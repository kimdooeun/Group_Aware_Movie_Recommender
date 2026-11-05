"""Group-level evaluation of the MF and content-based models on random virtual groups.

    python eval_group.py --mf-scores /path/to/mf_scores.csv

Groups of 2-5 users are sampled at random. Each model scores every candidate movie for every member
(MF: mf_scores.csv, content-based: content_based.score_user), member scores are aggregated into one
group score (Average / Least Misery) and the group's Top-K list is graded against each member's own
movies rated >= REL_THRESHOLD in ratings_test_eval.csv. mf.py / content_based.py are used as they are.

Outputs (results/group_eval/): group_eval_groups.csv, group_eval_by_group.csv, group_eval_summary.csv,
group_eval_diff.csv, group_eval_meta.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import content_based as cb
from mf import load_ratings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RESULTS, "group_eval")  # everything the group evaluation writes
SIZES = (2, 3, 4, 5)
REL_THRESHOLD = 4  # a test rating >= 4 counts as "liked"
METRICS = ["p_mean", "p_min", "ndcg_mean", "ndcg_min"]  # member mean = satisfaction, member min = fairness
AGGREGATORS = {
    "average": lambda S: S.mean(axis=0),
    "least_misery": lambda S: S.min(axis=0),
}
MODELS = ("mf", "content")
TIE_DECIMALS = 10  # aggregated scores equal up to this many decimals are treated as tied


def locate(sorted_ids, values, strict=True):
    """Positions of values in sorted_ids; -1 for missing (error if strict)."""
    pos = np.searchsorted(sorted_ids, values)
    pos_c = np.minimum(pos, len(sorted_ids) - 1)
    ok = sorted_ids[pos_c] == values
    if strict and not ok.all():
        raise ValueError("%d ids not found" % (~ok).sum())
    return np.where(ok, pos_c, -1)


def build_index(train, test):
    """Dense user x movie masks: seen (train history) and rel (liked in test_eval)."""
    user_ids = np.sort(train["userId"].unique())
    items = np.sort(train["movieId"].unique())  # candidate pool = movies MF knows
    seen = np.zeros((len(user_ids), len(items)), dtype=bool)
    seen[locate(user_ids, train["userId"].to_numpy()), locate(items, train["movieId"].to_numpy())] = True
    liked = test[test["rating"] >= REL_THRESHOLD]
    rel = np.zeros_like(seen)
    rel[locate(user_ids, liked["userId"].to_numpy()), locate(items, liked["movieId"].to_numpy())] = True
    return user_ids, items, seen, rel


def load_mf(path, user_ids, items, seen, test):
    """mf_scores.csv -> dense matrix (NaN where the pair is in train). Verifies the file against the split."""
    mat = np.full(seen.shape, np.nan)
    for chunk in pd.read_csv(path, chunksize=2_000_000, encoding="utf-8-sig"):
        mat[locate(user_ids, chunk["userId"].to_numpy()), locate(items, chunk["movieId"].to_numpy())] = \
            chunk["score"].to_numpy()
    if not np.array_equal(np.isnan(mat), seen):
        raise ValueError("mf_scores.csv does not cover exactly the non-train (user, movie) pairs")
    pred = mat[locate(user_ids, test["userId"].to_numpy()), locate(items, test["movieId"].to_numpy())]
    return mat, float(np.sqrt(np.mean((test["rating"].to_numpy() - pred) ** 2)))


def load_content(users, user_ids, items):
    """content_based.score_user for each needed user -> dense matrix over the candidate pool."""
    mat = np.full((len(user_ids), len(items)), np.nan)
    for n, u in enumerate(users, 1):
        df = cb.score_user(u, exclude_seen=False)
        col = locate(items, df["movieId"].to_numpy(), strict=False)
        ok = col >= 0
        mat[locate(user_ids, np.array([u]))[0], col[ok]] = df["score"].to_numpy()[ok]
        if n % 1000 == 0:
            print("  content-based scores: %d/%d users" % (n, len(users)))
    return mat


def check_content_reproduces_team_output():
    new = cb.score_user(user_id=1, top_n=5)
    old = pd.read_csv(os.path.join(RESULTS, "content_user1_top5.csv"))
    return bool(list(new["movieId"]) == list(old["movieId"]) and np.allclose(new["score"], old["score"]))


def sample_groups(user_ids, per_size, seed):
    rng = np.random.RandomState(seed)
    return [rng.choice(user_ids, size, replace=False) for size in SIZES for _ in range(per_size)]


def top_k(score, cand, k, tiekey):
    """Top-k column indices among candidates; ties broken by ascending tiekey. Also: is rank k tied with k+1?"""
    s = np.where(cand, np.round(score, TIE_DECIMALS), -np.inf)
    order = np.lexsort((tiekey, -s))
    return order[:k], bool(s[order[k - 1]] == s[order[k]])


def member_metrics(liked, n_liked, top, k):
    """Precision@k and binary-relevance NDCG@k per member (rows of liked: members x movies)."""
    hit = liked[:, top]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = np.concatenate([[0.0], np.cumsum(discount)])
    return hit.sum(axis=1) / k, (hit * discount).sum(axis=1) / idcg[np.minimum(n_liked, k)]


def evaluate_groups(groups, scores, user_ids, seen, rel, k, tie_seeds, seed):
    n_liked_all = rel.sum(axis=1)
    n_items = seen.shape[1]
    rows, n_members, n_excluded, n_skipped = [], 0, 0, 0
    for gid, members in enumerate(groups):
        r = locate(user_ids, members)
        valid = n_liked_all[r] > 0
        n_members += len(r)
        n_excluded += int((~valid).sum())
        if not valid.any():
            n_skipped += 1
            continue
        cand = ~seen[r].any(axis=0)
        liked, n_liked = rel[r[valid]], n_liked_all[r[valid]]
        by_id = np.arange(n_items)  # ascending movieId == content_based.score_user's own tie rule
        perms = [np.random.default_rng([seed, gid, s]).permutation(n_items) for s in range(tie_seeds)]

        for model in MODELS:
            for agg_name, agg in AGGREGATORS.items():
                group_score = agg(scores[model][r])
                base = {"group_id": gid, "size": len(r), "model": model, "agg": agg_name, "n_graded": int(valid.sum())}

                top, tie = top_k(group_score, cand, k, by_id)
                p, n = member_metrics(liked, n_liked, top, k)
                rows.append({**base, "tie": "movieId", "tie_at_cutoff": float(tie),
                             "p_mean": p.mean(), "p_min": p.min(), "ndcg_mean": n.mean(), "ndcg_min": n.min()})

                per_seed = []
                for perm in perms:
                    top, _ = top_k(group_score, cand, k, perm)
                    p, n = member_metrics(liked, n_liked, top, k)
                    per_seed.append([p.mean(), p.min(), n.mean(), n.min()])
                rows.append({**base, "tie": "random", "tie_at_cutoff": np.nan,
                             **dict(zip(METRICS, np.mean(per_seed, axis=0)))})
    return pd.DataFrame(rows), n_members, n_excluded, n_skipped


def summarize(df):
    df = df.assign(size=df["size"].astype(str))
    both = pd.concat([df, df.assign(size="all")])
    keys = ["size", "model", "agg", "tie"]
    out = both.groupby(keys)[METRICS + ["tie_at_cutoff"]].mean()
    out["n_groups"] = both.groupby(keys).size()
    return out.reset_index()


def bootstrap_diff(df, n_boot=2000, seed=42):
    """Mean (MF - content) per metric with a paired bootstrap 95% CI over groups."""
    df = df.assign(size=df["size"].astype(str))
    both = pd.concat([df, df.assign(size="all")])
    idx = ["group_id", "size", "agg", "tie"]
    mf = both[both["model"] == "mf"].set_index(idx)[METRICS]
    ct = both[both["model"] == "content"].set_index(idx)[METRICS]
    diff = (mf - ct).reset_index()
    rng = np.random.RandomState(seed)
    out = []
    for (size, agg, tie), g in diff.groupby(["size", "agg", "tie"]):
        pick = rng.randint(0, len(g), (n_boot, len(g)))
        for m in METRICS:
            d = g[m].to_numpy()
            boot = d[pick].mean(axis=1)
            out.append({"size": size, "agg": agg, "tie": tie, "metric": m, "n_groups": len(g),
                        "mf_minus_content": d.mean(), "ci_low": np.percentile(boot, 2.5),
                        "ci_high": np.percentile(boot, 97.5)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mf-scores", default=os.path.join(RESULTS, "mf_scores.csv"))
    ap.add_argument("--groups-per-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tie-seeds", type=int, default=5)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    t0 = time.time()

    print("[1/5] load split and build masks")
    train = load_ratings(os.path.join(RESULTS, "ratings_train.csv"))
    test = load_ratings(os.path.join(RESULTS, "ratings_test_eval.csv"))
    user_ids, items, seen, rel = build_index(train, test)
    print("  users %d, candidate movies %d, users without any liked test movie %d"
          % (len(user_ids), len(items), int((rel.sum(axis=1) == 0).sum())))

    print("[2/5] sample groups (seed %d)" % args.seed)
    groups = sample_groups(user_ids, args.groups_per_size, args.seed)
    needed = np.unique(np.concatenate(groups))
    print("  %d groups, %d distinct users" % (len(groups), len(needed)))

    print("[3/5] load model scores")
    mf_mat, mf_rmse = load_mf(args.mf_scores, user_ids, items, seen, test)
    expected = float(pd.read_csv(os.path.join(RESULTS, "mf_final.csv"))["test_rmse"].iloc[0])
    print("  MF test RMSE from scores file %.4f (team report %.4f)" % (mf_rmse, expected))
    if abs(mf_rmse - expected) > 1e-3:
        raise ValueError("mf_scores.csv does not reproduce the reported MF test RMSE")
    cb_ok = check_content_reproduces_team_output()
    print("  content_based.score_user(1, top_n=5) matches content_user1_top5.csv:", cb_ok)
    if not cb_ok:
        raise ValueError("content-based model does not reproduce the team output")
    scores = {"mf": mf_mat, "content": load_content(needed, user_ids, items)}

    print("[4/5] evaluate (K=%d)" % args.k)
    df, n_members, n_excluded, n_skipped = evaluate_groups(
        groups, scores, user_ids, seen, rel, args.k, args.tie_seeds, args.seed)
    print("  members %d, excluded (no liked test movie) %d, groups skipped %d" % (n_members, n_excluded, n_skipped))

    print("[5/5] summarize and save")
    summary, diff = summarize(df), bootstrap_diff(df)
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame({"group_id": range(len(groups)), "size": [len(g) for g in groups],
                  "userIds": ["|".join(map(str, g)) for g in groups]}
                 ).to_csv(os.path.join(args.out_dir, "group_eval_groups.csv"), index=False)
    df.to_csv(os.path.join(args.out_dir, "group_eval_by_group.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "group_eval_summary.csv"), index=False)
    diff.to_csv(os.path.join(args.out_dir, "group_eval_diff.csv"), index=False)
    with open(os.path.join(args.out_dir, "group_eval_meta.json"), "w") as f:
        json.dump({"seed": args.seed, "groups_per_size": args.groups_per_size, "k": args.k,
                   "rel_threshold": REL_THRESHOLD, "tie_seeds": args.tie_seeds,
                   "candidate_movies": len(items), "mf_test_rmse_from_scores_file": mf_rmse,
                   "content_reproduces_team_output": cb_ok, "members": n_members,
                   "members_excluded": n_excluded, "groups_skipped": n_skipped}, f, indent=2)

    pd.set_option("display.width", 200)
    print(summary[summary["size"] == "all"].round(4).to_string(index=False))
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
