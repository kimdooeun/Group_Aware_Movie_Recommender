# -*- coding: utf-8 -*-
"""Amazon MF + Content-based group Top-10, matched to eval_group_knn.py.

Uses the algorithm of mf.py and feature construction of content_based.py,
but loads Amazon CSVs explicitly. Same groups/candidate pool/top-k/metrics as KNN.

Run from ML root:
  python eval_amazon_mf_content.py --data-dir results/amazon --knn-dir results/amazon/knn_baseline

Fast optional COMMON-CANDIDATE experiment: first make the candidate file and re-run
KNN with exactly the same candidate file/groups, then run this script with it.
"""
import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from mf import MatrixFactorization
from eval_group_knn import (
    METRICS, load_split, load_d_groups, check_groups, get_candidate_mask,
    positives_from_test, group_metric, top_k_like_d, summarize,
)


def choose_popular_candidates(train, n, out_path):
    # Use TRAIN only; not ratings from test.
    counts = train.groupby('movieId').size().reset_index(name='train_count')
    counts = counts.sort_values(['train_count', 'movieId'], ascending=[False, True])
    out = counts.head(n)[['movieId', 'train_count']]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'후보 {len(out):,}개 저장: {out_path}')


def load_content_features(meta_path, item_ids):
    if not meta_path.is_file():
        raise FileNotFoundError(f'Amazon 콘텐츠 CSV가 없습니다: {meta_path}')
    meta = pd.read_csv(meta_path, encoding='utf-8-sig',
                       usecols=['movieId', 'genres', 'keywords', 'overview'])
    if meta.movieId.duplicated().any():
        raise ValueError('Amazon 메타데이터에 중복 movieId가 있습니다.')
    # Exact candidate ITEM order of MF/KNN; blanks are legitimate missing metadata.
    meta = meta.set_index('movieId').reindex(item_ids)
    for key in ('genres', 'keywords', 'overview'):
        meta[key] = meta[key].fillna('').astype(str)
    # C's content_based.py repeats genres three times, then adds keywords/overview.
    genre_text = meta['genres'].str.replace('|', ' ', regex=False)
    text = ((genre_text + ' ') * 3 + meta['keywords'] + ' ' + meta['overview']).tolist()
    del meta, genre_text
    vec = TfidfVectorizer(stop_words='english', max_features=10000, dtype=np.float32)
    try:
        X = vec.fit_transform(text).tocsr()
    except ValueError as e:
        raise ValueError('TF-IDF가 비었습니다. amazon_movies_processed.csv의 텍스트를 확인하세요.') from e
    print(f'    TF-IDF 형태: {X.shape}, 저장된 비영 값: {X.nnz:,}', flush=True)
    return X


def make_train_history(train, user_ids, item_ids):
    # Single source of truth for group-unseen pool (same as D and KNN).
    cols = np.searchsorted(item_ids, train.movieId.to_numpy(dtype=np.int64))
    users = train.userId.to_numpy(dtype=np.int64)
    ratings = train.rating.to_numpy(dtype=np.float32)
    by_user = {}
    for user in np.unique(users):
        by_user[int(user)] = None
    # Stable ordering not essential; avoid per-user dataframe filtering across groups.
    order = np.argsort(users, kind='stable')
    us, cs, rs = users[order], cols[order], ratings[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(us)) + 1, len(us)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        by_user[int(us[start])] = (cs[start:end], rs[start:end])
    return by_user


class ContentProfiles:
    def __init__(self, X, by_user, cache_size=32):
        self.X = X
        self.history = by_user
        self.profile = lru_cache(maxsize=cache_size)(self._compute)

    def _compute(self, uid):
        columns, ratings = self.history[int(uid)]
        weights = ratings - 3.0  # same as C's score_user()
        profile = np.asarray(self.X[columns].multiply(weights[:, None]).sum(axis=0)).ravel()
        length = np.linalg.norm(profile)
        if length > 0:
            profile /= length
        return profile.astype(np.float32, copy=False)

    def group_scores(self, members, columns):
        profiles = np.stack([self.profile(int(uid)) for uid in members], axis=0)
        # TF-IDF rows have unit norm; normalized user profile => cosine similarity.
        out = self.X[columns].dot(profiles.T)
        return np.asarray(out, dtype=np.float64).T


def mf_scores(model, user_idx, columns):
    # Only score needed group members and common candidate items. No giant score CSV.
    scores = (model.mu + model.bu[user_idx, None]
              + model.bi[columns][None, :]
              + model.P[user_idx] @ model.Q[columns].T)
    return np.clip(scores, 1.0, 5.0)


def compare_to_knn(knn_dir, groups, by_group, out_dir, topk, threshold, candidate_file):
    groupfile = knn_dir / 'group_eval_groups.csv'
    byfile = knn_dir / 'group_eval_by_group.csv'
    metadata = knn_dir / 'knn_meta.json'
    if not groupfile.is_file() or not byfile.is_file() or not metadata.is_file():
        raise FileNotFoundError('3모델 비교에는 KNN의 group_eval_groups.csv, '
                                'group_eval_by_group.csv, knn_meta.json 세 파일이 필요합니다.')
    knn_groups = load_d_groups(groupfile)
    knn_lookup = {gid: members for gid, members in knn_groups}
    for gid, members in groups:
        if gid not in knn_lookup or members != knn_lookup[gid]:
            raise ValueError(f'KNN과 그룹 {gid}의 구성원이 다릅니다.')
    cfg = json.loads(metadata.read_text(encoding='utf-8'))
    if cfg.get('top_k') != topk or cfg.get('rel_threshold') != threshold:
        raise ValueError('KNN과 Top-K 또는 정답 평점 기준이 다릅니다.')
    knn_candidate = cfg.get('candidate_file')
    if (knn_candidate is None) != (candidate_file is None):
        raise ValueError('KNN과 후보 범위가 다릅니다. 후보 파일을 설정했다면 KNN도 다시 실행하세요.')
    if candidate_file is not None and Path(knn_candidate).resolve() != Path(candidate_file).resolve():
        raise ValueError('KNN과 --candidate-file 경로가 다릅니다. 동일한 파일을 지정하세요.')
    kdf = pd.read_csv(byfile, encoding='utf-8-sig')
    kdf = kdf[(kdf.model == 'knn') & (kdf.group_id.isin([gid for gid, _ in groups]))]
    if kdf.empty:
        raise ValueError('KNN 그룹별 결과에 해당 group_id가 없습니다.')
    # Same valid groups per model; check rather than hide unequal group samples.
    knn_ids = set(kdf.group_id.astype(int))
    model_ids = set(by_group.group_id.astype(int))
    if knn_ids != model_ids:
        raise ValueError(f'KNN과 MF/Content 평가 가능한 그룹이 다릅니다: '
                         f'{len(knn_ids)} vs {len(model_ids)}')
    joined = pd.concat([kdf, by_group], ignore_index=True)
    joined.to_csv(out_dir / 'group_eval_all_three_by_group.csv', index=False, encoding='utf-8-sig')
    combined = summarize(joined)
    combined.to_csv(out_dir / 'group_eval_all_three_summary.csv', index=False, encoding='utf-8-sig')
    return combined


def main():
    ap = argparse.ArgumentParser(description='Amazon MF/Content group evaluation with existing KNN groups')
    ap.add_argument('--data-dir', default='results/amazon')
    ap.add_argument('--knn-dir', default='results/amazon/knn_baseline')
    ap.add_argument('--groups-file', default=None)
    ap.add_argument('--out-dir', default='results/amazon/mf_content_eval')
    ap.add_argument('--candidate-file', default=None)
    ap.add_argument('--make-top-candidates', type=int, default=None,
                    help='TRAIN 평점 수가 많은 N개 상품 후보 CSV만 생성하고 종료')
    ap.add_argument('--max-groups', type=int, default=None, help='시범 평가할 그룹 수 제한')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--threshold', type=float, default=4.0)
    ap.add_argument('--tie-seeds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mf-k', type=int, default=50)
    ap.add_argument('--mf-lr', type=float, default=0.01)
    ap.add_argument('--mf-reg', type=float, default=0.05)
    ap.add_argument('--mf-epochs', type=int, default=20)
    ap.add_argument('--print-every', type=int, default=100)
    ap.add_argument('--skip-combine', action='store_true')
    args = ap.parse_args()
    if args.k < 1 or args.tie_seeds < 0 or args.mf_epochs < 1:
        ap.error('k>=1, tie-seeds>=0, mf-epochs>=1 이어야 합니다.')
    t0 = time.perf_counter()
    base = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    knn_dir = Path(args.knn_dir)
    print('[1/5] Amazon 평가 데이터 읽기', flush=True)
    train, test = load_split(base, 'amazon')
    if args.make_top_candidates:
        choose_popular_candidates(train, args.make_top_candidates, base / 'amazon_common_candidates.csv')
        return
    user_ids = np.sort(train.userId.unique().astype(np.int64))
    item_ids = np.sort(train.movieId.unique().astype(np.int64))
    groups_path = Path(args.groups_file) if args.groups_file else knn_dir / 'group_eval_groups.csv'
    groups = load_d_groups(groups_path)
    check_groups(groups, user_ids)
    if args.max_groups is not None:
        groups = groups[:args.max_groups]
    if not groups:
        raise ValueError('평가할 그룹이 없습니다.')
    common_mask = get_candidate_mask(Path(args.candidate_file) if args.candidate_file else None,
                                     item_ids, False)
    history = make_train_history(train, user_ids, item_ids)
    liked = positives_from_test(test, args.threshold)
    print(f'    그룹 {len(groups):,}개 / 전체 학습 상품 {len(item_ids):,}개 / '
          f'공통 후보 {common_mask.sum():,}개', flush=True)

    print('[2/5] Amazon TRAIN으로 팀 MF 알고리즘 새로 학습', flush=True)
    mf = MatrixFactorization(k=args.mf_k, lr=args.mf_lr, reg=args.mf_reg,
                             epochs=args.mf_epochs, seed=args.seed)
    # Never feed evaluation holdout into training/selection.
    mf.fit(train, val=None, verbose=True)
    mf_index = {int(uid): i for i, uid in enumerate(mf.user_ids)}
    if not np.array_equal(mf.item_ids, item_ids):
        raise ValueError('MF와 평가의 상품 ID 배열이 다릅니다.')

    print('[3/5] Amazon 메타데이터로 팀 Content-based TF-IDF 생성', flush=True)
    X = load_content_features(base / 'amazon_movies_processed.csv', item_ids)
    profiles = ContentProfiles(X, history)

    print('[4/5] 동일한 그룹과 후보에서 MF / Content 그룹 추천 평가', flush=True)
    discounts = 1 / np.log2(np.arange(2, args.k + 2))
    idcg = np.r_[0, np.cumsum(discounts)]
    rows, tops = [], []
    skipped = 0
    for group_index, (gid, members) in enumerate(groups):
        valid = [u for u in members if liked.get(u)]
        if not valid:
            skipped += 1
            continue
        keep = common_mask.copy()
        for uid in members:
            keep[history[int(uid)][0]] = False
        cols = np.flatnonzero(keep)
        if len(cols) < args.k:
            raise ValueError(f'그룹 {gid}: 후보 {len(cols)}개로 Top-{args.k} 불가능')
        mf_user_rows = np.array([mf_index[int(uid)] for uid in members])
        individual_scores = {
            'mf': mf_scores(mf, mf_user_rows, cols),
            'content': profiles.group_scores(members, cols),
        }
        priorities = cols
        random_priorities = None
        for model_name, scores in individual_scores.items():
            for agg in ('average', 'least_misery'):
                vals = scores.mean(axis=0) if agg == 'average' else scores.min(axis=0)
                selected, tied = top_k_like_d(vals, priorities, args.k)
                mids = item_ids[cols[selected]]
                grades = group_metric(members, mids, liked, args.k, discounts, idcg)
                base_row = {'group_id': gid, 'size': len(members), 'model': model_name,
                            'agg': agg, 'n_graded': grades['n_graded']}
                rows.append({**base_row, 'tie': 'movieId', 'tie_at_cutoff': float(tied),
                             **{m: grades[m] for m in METRICS}})
                for rank, p in enumerate(selected, 1):
                    tops.append({'group_id': gid, 'model': model_name, 'agg': agg,
                                 'rank': rank, 'movieId': int(item_ids[cols[p]]),
                                 'group_score': float(vals[p])})
                if args.tie_seeds > 0:
                    if tied:
                        if random_priorities is None:
                            # Same RNG construction as D and eval_group_knn.py.
                            random_priorities = [
                                np.random.default_rng([args.seed, group_index, s]).permutation(len(item_ids))[cols]
                                for s in range(args.tie_seeds)
                            ]
                        trials = []
                        for tiekey in random_priorities:
                            selected_rand, _ = top_k_like_d(vals, tiekey, args.k)
                            g = group_metric(members, item_ids[cols[selected_rand]], liked,
                                             args.k, discounts, idcg)
                            trials.append([g[m] for m in METRICS])
                        random_mean = np.mean(trials, axis=0)
                    else:
                        random_mean = [grades[m] for m in METRICS]
                    rows.append({**base_row, 'tie': 'random', 'tie_at_cutoff': np.nan,
                                 **dict(zip(METRICS, map(float, random_mean)))})
        if args.print_every and (group_index + 1) % args.print_every == 0:
            print(f'    {group_index+1:,}/{len(groups):,} 그룹 완료', flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'group_eval_mf_content_by_group.csv', index=False, encoding='utf-8-sig')
    smry = summarize(df)
    smry.to_csv(out_dir / 'group_eval_mf_content_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(tops).to_csv(out_dir / 'mf_content_top10.csv', index=False, encoding='utf-8-sig')
    metadata = {'data_dir': str(base), 'groups_file': str(groups_path),
                'n_requested_groups': len(groups), 'n_evaluated_groups': df.group_id.nunique(),
                'skipped_groups': skipped, 'k': args.k, 'threshold': args.threshold,
                'tie_seeds': args.tie_seeds, 'candidate_file': args.candidate_file,
                'tfidf_max_features': 10000, 'mf': {'k': args.mf_k, 'lr': args.mf_lr,
                'reg': args.mf_reg, 'epochs': args.mf_epochs},
                'notes': 'MF fitted only with Amazon TRAIN. TF-IDF and user profiles built from Amazon metadata and TRAIN.'}
    (out_dir / 'mf_content_meta.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[5/5] Amazon 모델 성능 요약', flush=True)
    print(smry[(smry['size'] == 'all') & (smry['tie'] == 'movieId')]
          [['model', 'agg', 'p_mean', 'ndcg_mean', 'p_min', 'ndcg_min', 'n_groups']]
          .to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    if not args.skip_combine:
        combined = compare_to_knn(knn_dir, groups, df, out_dir, args.k,
                                  args.threshold, args.candidate_file)
        print('\n=== Amazon MF / Content-based / kNN, 같은 그룹 비교 ===')
        print(combined[(combined['size'] == 'all') & (combined['tie'] == 'movieId')]
              [['model', 'agg', 'p_mean', 'ndcg_mean', 'p_min', 'ndcg_min', 'n_groups']]
              .to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print(f'완료: {out_dir.resolve()} / {time.perf_counter()-t0:.1f}초', flush=True)


if __name__ == '__main__':
    main()
