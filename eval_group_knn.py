# -*- coding: utf-8 -*-
"""User-based kNN group baseline aligned with the team's eval_group.py.

MovieLens:
  python eval_group_knn.py --dataset movielens --data-dir results \
      --groups-file results/group_eval/group_eval_groups.csv \
      --compare-dir results/group_eval
Amazon:
  python eval_group_knn.py --dataset amazon --data-dir results/amazon

Requires: numpy, pandas, scipy. Does NOT import MF or content_based.
The SAME evaluation rules are used in both domains, and the recommender is
fitted independently to each dataset's TRAIN ratings only.

Correspondence with eval_group.py:
- eg.sample_groups: np.random.RandomState(seed), sizes 2/3/4/5 in that order,
  1000 groups/size by default, IDs starting at 0.
- eg.build_index: candidates are only TRAIN items, unseen by ALL group members.
- eg.AGGREGATORS: average and least_misery of individual predicted ratings.
- eg.top_k: round to 10 decimals; ascending movieId to resolve ties; random
  tie priorities from default_rng([seed, group_index, tie_seed]).permutation.
- eg.member_metrics: positives are TEST_EVAL ratings >=4; IDCG is based on the
  user's ENTIRE positive test count, including positives excluded by group pool.
- eg.evaluate_groups: no-positive members participate in recommendation but
  not its member-level metric; p_mean, p_min, ndcg_mean, ndcg_min.

Method-specific additions (not claimed to be part of D's implementation):
- k=30 nearest users, positively correlated mean-centered cosine similarity,
  optional similarity shrinkage by the number of common TRAIN-rated items.
- If no positive neighbor rated item i, the smoothed TRAIN item mean shifted by
  the target user's average-rating tendency is used as a fallback.
- The sparse item->user index avoids an all-user x all-user similarity matrix.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

SIZES = (2, 3, 4, 5)
METRICS = ["p_mean", "p_min", "ndcg_mean", "ndcg_min"]
AGGS = ("average", "least_misery")
EPS = 1e-12


def load_split(folder: Path, dataset: str):
    prefix = "ratings" if dataset == "movielens" else "amazon"
    paths = [folder / (prefix + name + ".csv") for name in ("_train", "_test_eval")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"파일이 없습니다: {path}\n--data-dir 위치를 확인하세요.")
    cols = ["userId", "movieId", "rating"]
    train = pd.read_csv(paths[0], encoding="utf-8-sig", usecols=cols)
    test = pd.read_csv(paths[1], encoding="utf-8-sig", usecols=cols)
    for name, frame in (("train", train), ("test_eval", test)):
        if frame.empty or frame[cols].isna().any().any():
            raise ValueError(f"{name} 데이터가 비었거나 ID/평점 결측치가 있습니다.")
        for col in ("userId", "movieId"):
            frame[col] = pd.to_numeric(frame[col], errors="raise").astype(np.int64)
        frame["rating"] = pd.to_numeric(frame["rating"], errors="raise")
        if not frame.rating.between(1, 5).all():
            raise ValueError(f"{name} 평점이 1~5 범위를 벗어납니다.")
        if frame.duplicated(["userId", "movieId"]).any():
            raise ValueError(f"{name}에 같은 (userId, movieId)가 중복됩니다.")
    # D's build_index requires all test IDs to be present in the TRAIN pool.
    for col in ("userId", "movieId"):
        extra = np.setdiff1d(test[col].unique(), train[col].unique())
        if extra.size:
            raise ValueError(
                f"test_eval에서 학습에 없는 {col} {extra.size:,}개를 발견했습니다. "
                "D의 eval_group.py와 동일하게 평가하려면 학습에 존재하는 ID만 남긴 "
                f"{prefix}_test_eval.csv를 사용하세요. 예: {extra[:5].tolist()}"
            )
    shared = pd.MultiIndex.from_frame(train[["userId", "movieId"]]).intersection(
        pd.MultiIndex.from_frame(test[["userId", "movieId"]])
    )
    if len(shared):
        raise ValueError("TRAIN과 TEST_EVAL에 같은 (userId, movieId)가 있습니다. 평가 누출입니다.")
    return train, test


def d_sample_groups(user_ids, groups_per_size: int, seed: int):
    """Precisely the same RNG and iteration as eval_group.sample_groups."""
    rng = np.random.RandomState(seed)
    return [(gid, rng.choice(user_ids, size, replace=False).astype(int).tolist())
            for gid, size in enumerate(s for s in SIZES for _ in range(groups_per_size))]


def load_d_groups(path: Path):
    """Read D's group_eval_groups.csv: group_id,size,userIds ('1|2|3')."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "group_id" not in frame.columns:
        raise ValueError(f"{path}: group_id 열이 필요합니다.")
    if "userIds" in frame.columns:
        groups = []
        for row in frame.itertuples(index=False):
            raw = getattr(row, "userIds")
            members = [int(v) for v in str(raw).replace(",", "|").split("|") if v.strip()]
            groups.append((int(getattr(row, "group_id")), members))
    elif "userId" in frame.columns:
        groups = [(int(gid), part["userId"].astype(int).tolist())
                  for gid, part in frame.groupby("group_id", sort=False)]
    else:
        raise ValueError(f"{path}: userIds 또는 userId 열이 필요합니다.")
    if "size" in frame.columns and "userIds" in frame.columns:
        for (gid, mem), stated in zip(groups, frame["size"]):
            if len(mem) != int(stated):
                raise ValueError(f"그룹 {gid}: size와 userIds가 불일치합니다.")
    if len({x for x, _ in groups}) != len(groups):
        raise ValueError("그룹 ID가 중복되어 있습니다.")
    return groups


def check_groups(groups, user_ids):
    known = set(map(int, user_ids))
    for gid, members in groups:
        if len(members) not in SIZES or len(set(members)) != len(members):
            raise ValueError(f"그룹 {gid}: 서로 다른 2~5명의 사용자여야 합니다.")
        unknown = set(members) - known
        if unknown:
            raise ValueError(f"그룹 {gid}: train에 없는 사용자 {sorted(unknown)}")


def get_candidate_mask(path: Path | None, item_ids: np.ndarray, require_text: bool):
    mask = np.ones(len(item_ids), dtype=bool)
    if path is None:
        if require_text:
            raise ValueError("--require-text는 --candidate-file과 함께 지정해야 합니다.")
        return mask
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "movieId" not in frame:
        raise ValueError("후보 CSV에는 movieId 열이 필요합니다.")
    if require_text:
        if "has_content_text" in frame:
            accepted = frame["has_content_text"].astype(str).str.strip().str.lower().isin(
                ("true", "1", "yes"))
        elif {"genres", "keywords", "overview"}.issubset(frame.columns):
            accepted = frame[["genres", "keywords", "overview"]].fillna("").astype(str).apply(
                lambda col: col.str.strip().ne("")).any(axis=1)
        else:
            raise ValueError("--require-text에 필요한 has_content_text 또는 텍스트 열이 없습니다.")
        frame = frame.loc[accepted]
    allow = pd.to_numeric(frame.movieId, errors="raise").astype(np.int64).unique()
    mask = np.isin(item_ids, allow)
    if not mask.any():
        raise ValueError("후보 파일과 TRAIN의 movieId가 일치하는 상품이 없습니다.")
    return mask


class UserKNN:
    """User-user centered cosine kNN with sparse inverted-index neighbor lookup."""

    def __init__(self, train: pd.DataFrame, k: int = 30, shrink: float = 10.0,
                 min_overlap: int = 1, prior: float = 20.0, cache_size: int = 24):
        if k < 1 or min_overlap < 1 or shrink < 0 or prior < 0 or cache_size < 0:
            raise ValueError("k, min_overlap >= 1; shrink, prior, cache_size >= 0이어야 합니다.")
        self.k = int(k)
        self.shrink = float(shrink)
        self.min_overlap = int(min_overlap)
        self.user_ids = np.sort(train.userId.unique().astype(np.int64))
        self.item_ids = np.sort(train.movieId.unique().astype(np.int64))
        if len(self.user_ids) < 5 or len(self.item_ids) < 11:
            raise ValueError("데이터에 최소 사용자 5명, 아이템 11개가 필요합니다.")
        self.u_index = {int(val): ix for ix, val in enumerate(self.user_ids)}
        u = np.searchsorted(self.user_ids, train.userId.to_numpy(dtype=np.int64)).astype(np.int32)
        i = np.searchsorted(self.item_ids, train.movieId.to_numpy(dtype=np.int64)).astype(np.int32)
        r = train.rating.to_numpy(dtype=np.float32)
        shape = (len(self.user_ids), len(self.item_ids))
        self.R = csr_matrix((r, (u, i)), shape=shape, dtype=np.float32)
        counts = np.bincount(u, minlength=shape[0]); totals = np.bincount(u, weights=r, minlength=shape[0])
        self.means = totals / np.maximum(counts, 1)
        self.global_mean = float(np.mean(r))
        # Keep stored zeros so common-rating overlap counts include ratings equal to the user's mean.
        deviations = r.astype(np.float64) - self.means[u]
        C = csr_matrix((deviations, (u, i)), shape=shape, dtype=np.float64)
        norms = np.sqrt(np.asarray(C.multiply(C).sum(axis=1)).ravel())
        inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > EPS)
        C.data *= np.repeat(inv, np.diff(C.indptr))
        self.normalized = C.tocsr()
        self.by_item = C.tocsc()
        totals_i = np.bincount(i, weights=r, minlength=shape[1]); counts_i = np.bincount(i, minlength=shape[1])
        self.smoothed_item_mean = (totals_i + prior * self.global_mean) / (counts_i + prior)
        self.zero_norm_users = int((norms <= EPS).sum())
        self.neighbor_queries = 0
        self.with_neighbors = 0
        self.neighbor_item_coverage = 0.0
        self.score_user = lru_cache(maxsize=cache_size)(self._score_user)

    def neighbors(self, user_id: int):
        ix = self.u_index[int(user_id)]
        ptr = self.normalized.indptr
        rated_cols = self.normalized.indices[ptr[ix]:ptr[ix + 1]]
        weights = self.normalized.data[ptr[ix]:ptr[ix + 1]]
        self.neighbor_queries += 1
        if not np.any(np.abs(weights) > EPS):
            return np.empty(0, np.int32), np.empty(0, np.float64)
        # Only visit users who share a TRAIN item (not every other user).
        index_parts, value_parts = [], []
        colptr = self.by_item.indptr
        for item, w in zip(rated_cols, weights):
            a, b = colptr[item], colptr[item + 1]
            index_parts.append(self.by_item.indices[a:b])
            value_parts.append(self.by_item.data[a:b] * w)
        if not index_parts:
            return np.empty(0, np.int32), np.empty(0, np.float64)
        nbr_rows = np.concatenate(index_parts)
        dots = np.bincount(nbr_rows, weights=np.concatenate(value_parts), minlength=len(self.user_ids))
        overlaps = np.bincount(nbr_rows, minlength=len(self.user_ids))
        candidates = np.flatnonzero((dots > EPS) & (overlaps >= self.min_overlap))
        candidates = candidates[candidates != ix]
        if not len(candidates):
            return np.empty(0, np.int32), np.empty(0, np.float64)
        sim = dots[candidates]
        if self.shrink:
            sim *= overlaps[candidates] / (overlaps[candidates] + self.shrink)
        order = np.lexsort((self.user_ids[candidates], -sim))[:self.k]
        self.with_neighbors += 1
        return candidates[order].astype(np.int32), sim[order]

    def _score_user(self, user_id: int) -> np.ndarray:
        ix = self.u_index[int(user_id)]
        # Item-mean fallback is computed ONLY from TRAIN; no test information.
        pred = self.smoothed_item_mean.copy()
        pred += self.means[ix] - self.global_mean
        neighbors, similarities = self.neighbors(int(user_id))
        numerator = np.zeros(len(self.item_ids), dtype=np.float64)
        denominator = np.zeros(len(self.item_ids), dtype=np.float64)
        for v, sim in zip(neighbors, similarities):
            a, b = self.R.indptr[v], self.R.indptr[v + 1]
            idx = self.R.indices[a:b]
            rating = self.R.data[a:b]
            numerator[idx] += sim * (rating - self.means[v])
            denominator[idx] += sim
        covered = denominator > EPS
        pred[covered] = self.means[ix] + numerator[covered] / denominator[covered]
        self.neighbor_item_coverage += float(covered.mean())
        return np.clip(pred, 1.0, 5.0)

    def candidate_columns(self, members, global_mask):
        keep = global_mask.copy()
        for uid in members:
            ix = self.u_index[int(uid)]
            a, b = self.R.indptr[ix], self.R.indptr[ix + 1]
            keep[self.R.indices[a:b]] = False
        return np.flatnonzero(keep)


def top_k_like_d(values: np.ndarray, tie_priority: np.ndarray, k: int):
    """O(n) candidate pruning, EXACT lexsort ranking of eval_group.top_k.

    values already excludes non-candidates, so tie_priority is the corresponding
    original item-position priority (ascending movieId) or seeded permutation.
    """
    scores = np.round(np.asarray(values, dtype=np.float64), 10)
    if len(scores) < k:
        raise ValueError(f"그룹 후보가 {len(scores)}개인데 Top-{k}를 요청했습니다.")
    if not np.isfinite(scores).all():
        raise ValueError("추천 점수에 NaN 또는 무한대가 있습니다.")
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    higher = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    slots = k - len(higher)
    if len(tied) > slots:
        chosen = tied[np.argpartition(tie_priority[tied], slots - 1)[:slots]]
    else:
        chosen = tied
    selected = np.concatenate([higher, chosen])
    selected = selected[np.lexsort((tie_priority[selected], -scores[selected]))]
    return selected, bool(len(tied) > slots)


def positives_from_test(test: pd.DataFrame, cutoff: float):
    positives = test[test.rating >= cutoff]
    liked = {}
    for user, grp in positives.groupby("userId"):
        liked[int(user)] = set(map(int, grp.movieId))
    return liked


def group_metric(members, top_ids, liked, k, discount, idcg):
    valid = [u for u in members if len(liked.get(int(u), ())) > 0]
    if not valid:
        return None
    p, ndcg = [], []
    for uid in valid:
        positives = liked[int(uid)]
        hit = np.fromiter((float(int(mid) in positives) for mid in top_ids), dtype=np.float64, count=k)
        p.append(hit.mean())
        # D normalizes by ALL test positives, not merely the eligible candidates.
        ndcg.append(float(hit @ discount / idcg[min(len(positives), k)]))
    return {"p_mean": float(np.mean(p)), "p_min": float(np.min(p)),
            "ndcg_mean": float(np.mean(ndcg)), "ndcg_min": float(np.min(ndcg)),
            "n_graded": len(valid)}


def summarize(by_group):
    # Column names / meaning exactly match D's group_eval_summary.csv
    d = by_group.assign(size=by_group["size"].astype(str))
    both = pd.concat((d, d.assign(size="all")), ignore_index=True)
    keys = ["size", "model", "agg", "tie"]
    out = both.groupby(keys)[METRICS + ["tie_at_cutoff"]].mean()
    out["n_groups"] = both.groupby(keys).size()
    return out.reset_index()


def evaluate(groups, knn, liked, allowed_mask, top_k: int, tie_seeds: int, seed: int, print_every: int):
    """Exactly D's ranking/evaluation protocol, with UserKNN personal scores."""
    discount = 1.0 / np.log2(np.arange(2, top_k + 2))
    idcg = np.concatenate(([0.0], np.cumsum(discount)))
    rows, top_rows = [], []
    excluded, skipped = 0, 0
    n_items = len(knn.item_ids)
    for group_index, (gid, members) in enumerate(groups):
        valid = [u for u in members if len(liked.get(int(u), ())) > 0]
        excluded += len(members) - len(valid)
        if not valid:
            skipped += 1
            continue
        columns = knn.candidate_columns(members, allowed_mask)
        if len(columns) < top_k:
            raise ValueError(f"group_id={gid}: 후보 상품 {len(columns)}개, Top-{top_k} 불가. "
                             "후보 필터를 완화하거나 평가 K를 조정하세요.")
        individual = np.stack([knn.score_user(uid)[columns] for uid in members])
        aggregates = {"average": individual.mean(axis=0), "least_misery": individual.min(axis=0)}
        priorities = np.arange(n_items, dtype=np.int64)[columns]
        random_priorities = None  # generate only if cutoff ties can affect the list
        for agg, scores in aggregates.items():
            base = {"group_id": gid, "size": len(members), "model": "knn", "agg": agg,
                    "n_graded": len(valid)}
            selected, is_tied = top_k_like_d(scores, priorities, top_k)
            top_ids = knn.item_ids[columns[selected]]
            metric = group_metric(members, top_ids, liked, top_k, discount, idcg)
            rows.append({**base, "tie": "movieId", "tie_at_cutoff": float(is_tied),
                         **{key: metric[key] for key in METRICS}})
            for rank, col_pos in enumerate(selected, 1):
                top_rows.append({"group_id": gid, "size": len(members), "agg": agg, "rank": rank,
                                 "movieId": int(knn.item_ids[columns[col_pos]]),
                                 "group_score": float(scores[col_pos])})
            if tie_seeds > 0:
                if not is_tied:
                    # If rank K is NOT tied with K+1, random priorities cannot change Top-K.
                    # This is numerically identical to D's multiple random-tie evaluations.
                    mean_random = [metric[key] for key in METRICS]
                else:
                    if random_priorities is None:
                        random_priorities = [
                            np.random.default_rng([seed, group_index, t]).permutation(n_items)[columns]
                            for t in range(tie_seeds)]
                    trials = []
                    for perm in random_priorities:
                        chosen, _ = top_k_like_d(scores, perm, top_k)
                        trial_ids = knn.item_ids[columns[chosen]]
                        m = group_metric(members, trial_ids, liked, top_k, discount, idcg)
                        trials.append([m[key] for key in METRICS])
                    mean_random = np.mean(trials, axis=0)
                rows.append({**base, "tie": "random", "tie_at_cutoff": np.nan,
                             **dict(zip(METRICS, map(float, mean_random)))})
        if print_every > 0 and (group_index + 1) % print_every == 0:
            print(f"    {group_index + 1:,} / {len(groups):,} 그룹 처리 완료", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(top_rows), excluded, skipped


def compare_existing(existing_dir, groups, knn_group_rows, output_dir):
    """Join existing D's MF/Content results with matching kNN rows. Read-only on D's folder."""
    existing_dir = Path(existing_dir)
    group_path, scored_path = (existing_dir / "group_eval_groups.csv",
                               existing_dir / "group_eval_by_group.csv")
    if not group_path.exists() or not scored_path.exists():
        raise FileNotFoundError("D의 group_eval_groups.csv, group_eval_by_group.csv가 필요합니다: "
                                f"{existing_dir}")
    original_groups = load_d_groups(group_path)
    if len(groups) != len(original_groups) or any(
        gid != original_gid or list(map(int, mem)) != list(map(int, original_mem))
        for (gid, mem), (original_gid, original_mem) in zip(groups, original_groups)
    ):
        raise ValueError("D의 기존 그룹과 kNN 평가 그룹이 다릅니다. --groups-file "
                         f"{group_path} 를 지정해 재실행하세요.")
    other = pd.read_csv(scored_path, encoding="utf-8-sig")
    required = {"group_id", "size", "model", "agg", "tie", *METRICS}
    if not required.issubset(other.columns):
        raise ValueError("D의 group_eval_by_group.csv에 필수 컬럼이 없습니다.")
    both = pd.concat([other, knn_group_rows], ignore_index=True)
    both.to_csv(output_dir / "group_eval_with_knn_by_group.csv", index=False, encoding="utf-8-sig")
    combined = summarize(both)
    combined.to_csv(output_dir / "group_eval_with_knn_summary.csv", index=False, encoding="utf-8-sig")
    return combined


def main():
    parser = argparse.ArgumentParser(description="User-based kNN; D's eval_group.py compatible group evaluation")
    parser.add_argument("--dataset", required=True, choices=("movielens", "amazon"))
    parser.add_argument("--data-dir", default=None, help="MovieLens: results / Amazon: results/amazon")
    parser.add_argument("--output-dir", default=None, help="별도 결과 폴더 (기존 D 파일을 덮어쓰지 않음)")
    parser.add_argument("--groups-file", default=None, help="D의 group_eval_groups.csv를 그대로 재사용")
    parser.add_argument("--compare-dir", default=None, help="D의 group_eval_by_group.csv가 있는 폴더")
    parser.add_argument("--groups-per-size", type=int, default=1000, help="2/3/4/5명 각각 생성하는 그룹 개수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbors", type=int, default=30, help="사용자 최근접 이웃 수")
    parser.add_argument("--shrink", type=float, default=10.0, help="공통 평점 개수로 유사도 축소")
    parser.add_argument("--min-overlap", type=int, default=1, help="이웃 사용자 최소 공통 평점 개수")
    parser.add_argument("--prior", type=float, default=20.0, help="이웃이 평가하지 않은 상품의 평균 평점 수축 강도")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rel-threshold", type=float, default=4.0)
    parser.add_argument("--tie-seeds", type=int, default=5, help="D와 같은 동점 랜덤 처리 반복 수; 0=생략")
    parser.add_argument("--cache-size", type=int, default=24)
    parser.add_argument("--candidate-file", default=None,
                        help="모든 모델이 공유할 후보 movieId 목록 CSV (임의 필터는 비교 조건을 변경함)")
    parser.add_argument("--require-text", action="store_true", help="후보 파일의 has_content_text=true만 사용")
    parser.add_argument("--print-every", type=int, default=100)
    args = parser.parse_args()
    if args.groups_per_size < 1 or args.tie_seeds < 0 or args.top_k < 1:
        parser.error("groups-per-size, top-k >=1이고 tie-seeds >=0이어야 합니다.")
    dataset = args.dataset
    data_dir = Path(args.data_dir or ("results" if dataset == "movielens" else "results/amazon"))
    output_dir = Path(args.output_dir or ("results/knn_movielens" if dataset == "movielens"
                                         else "results/amazon/knn_baseline"))
    start = perf_counter()
    print(f"[1/4] {dataset}: 학습/평가 데이터 읽기", flush=True)
    train, test = load_split(data_dir, dataset)
    print(f"    train {len(train):,}건, test_eval {len(test):,}건")
    knn = UserKNN(train, k=args.neighbors, shrink=args.shrink, min_overlap=args.min_overlap,
                  prior=args.prior, cache_size=args.cache_size)
    print(f"    사용자 {len(knn.user_ids):,}명, 학습에 나온 상품 {len(knn.item_ids):,}개")
    if args.groups_file:
        groups = load_d_groups(Path(args.groups_file))
        group_source = str(args.groups_file)
    else:
        groups = d_sample_groups(knn.user_ids, args.groups_per_size, args.seed)
        group_source = f"D.sample_groups(seed={args.seed}, groups_per_size={args.groups_per_size})"
    check_groups(groups, knn.user_ids)
    mask = get_candidate_mask(Path(args.candidate_file) if args.candidate_file else None,
                              knn.item_ids, args.require_text)
    liked = positives_from_test(test, args.rel_threshold)
    print(f"[2/4] kNN 평가: {len(groups):,}개 그룹, Top-{args.top_k}, "
          f"후보 {int(mask.sum()):,}개 (그룹별 본 상품 추가 제외)", flush=True)
    by_group, top10, excluded, skipped = evaluate(
        groups, knn, liked, mask, args.top_k, args.tie_seeds, args.seed, args.print_every)
    if by_group.empty:
        raise ValueError("평가 가능한 그룹이 없습니다(모든 그룹원에게 긍정 TEST 평점이 없음).")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[3/4] D와 동일한 컬럼 이름으로 결과 저장", flush=True)
    group_df = pd.DataFrame({"group_id": [gid for gid, _ in groups],
                             "size": [len(members) for _, members in groups],
                             "userIds": ["|".join(map(str, members)) for _, members in groups]})
    group_df.to_csv(output_dir / "group_eval_groups.csv", index=False, encoding="utf-8-sig")
    by_group.to_csv(output_dir / "group_eval_by_group.csv", index=False, encoding="utf-8-sig")
    top10.to_csv(output_dir / "knn_top10.csv", index=False, encoding="utf-8-sig")
    summary = summarize(by_group)
    summary.to_csv(output_dir / "group_eval_summary.csv", index=False, encoding="utf-8-sig")
    # Keep a friendlier alias for the previous version of this team's kNN baseline.
    summary.to_csv(output_dir / "knn_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "dataset": dataset, "model": "user_based_knn", "train_ratings": len(train),
        "test_eval_ratings": len(test), "train_users": len(knn.user_ids), "train_items": len(knn.item_ids),
        "groups_source": group_source, "groups": len(groups), "seed": args.seed,
        "neighbors": args.neighbors, "shrink": args.shrink, "min_overlap": args.min_overlap,
        "prior": args.prior, "cache_size": args.cache_size, "top_k": args.top_k,
        "rel_threshold": args.rel_threshold, "tie_seeds": args.tie_seeds,
        "candidate_file": args.candidate_file, "require_text": args.require_text,
        "allowed_items": int(mask.sum()), "excluded_member_positions_no_test_positives": excluded,
        "skipped_groups_no_test_positives": skipped, "neighbor_queries": knn.neighbor_queries,
        "users_with_positive_neighbors": knn.with_neighbors, "zero_norm_users": knn.zero_norm_users,
        "mean_fraction_items_with_neighbor_predictions": (
            knn.neighbor_item_coverage / max(knn.neighbor_queries, 1)),
        "fallback": "TRAIN shrunk item mean + (target user TRAIN mean - global TRAIN mean)",
        "ndcg_idcg": "all test_eval positives per member (same as D; not candidate positives)",
        "notes": "Unobserved user-item ratings are not verified negatives; evaluated as non-hits."
    }
    with (output_dir / "knn_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("[4/4] 전체 요약", flush=True)
    print(summary[(summary["size"] == "all") & (summary["tie"] == "movieId")]
          [["model", "agg", "p_mean", "ndcg_mean", "p_min", "ndcg_min", "n_groups"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    if args.compare_dir:
        combined = compare_existing(args.compare_dir, groups, by_group, output_dir)
        print("\n=== D의 MF·Content-based와 kNN, 같은 그룹 비교 ===")
        print(combined[(combined["size"] == "all") & (combined["tie"] == "movieId")]
              [["model", "agg", "p_mean", "ndcg_mean", "p_min", "ndcg_min", "n_groups"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"완료: {output_dir.resolve()} / {perf_counter() - start:.1f}초", flush=True)


if __name__ == "__main__":
    main()
