"""Diagnostic for the group evaluation of MF: which part of the model produces the Top-10 lists?

    python eval_group_diag_mf.py --mf-scores /path/to/mf_scores.csv [--cache-dir DIR]

Retrains MF with the saved best parameters, splits its score into bias and latent parts, grades the same 4,000
virtual groups exactly like eval_group.py (same candidates, Average / Least Misery, ties by ascending movieId)
and compares with non-personalised lists (shrunk item mean, item bias only) and with MF restricted to movies
with at least N train ratings. Also reports list diversity and the RMSE of MF, a bias-only model and the
global mean. --cache-dir only stores the fitted models so a rerun can skip the slow pure-Python training.

Output (results/group_eval/): group_eval_diag_mf.csv
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

import eval_group as eg
from mf import MatrixFactorization

RESULTS = eg.RESULTS
OUT = eg.OUT_DIR  # group_eval_* files live here
PRIOR = 10  # pseudo-ratings of the global mean in the shrunk item mean
MIN_COUNTS = (50, 100, 200)
SIZE_LABELS = ("all",) + eg.SIZES
GRADE_COLS = ["p_mean", "ndcg_mean", "p_min", "ndcg_min"]
COLUMNS = ["system", "agg", "size"] + GRADE_COLS + ["distinct_movies", "top_movie_share", "top_movie_title",
                                                    "mean_pair_overlap", "rmse"]


def fit_cached(name, cache_dir, train, test, **params):
    """Fit MatrixFactorization(**params) like run_mf.final(); with a cache dir, reuse/store the fitted model."""
    path = os.path.join(cache_dir, name + ".pkl") if cache_dir else None
    if path and os.path.exists(path):
        print("  %s: loaded from %s" % (name, path))
        with open(path, "rb") as f:
            return pickle.load(f)
    model = MatrixFactorization(**params)
    model.fit(train, test)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
    return model


def test_rmse(model, test):
    return model.evaluate(test)["rmse"]


def global_mean_rmse(train, test):
    return float(np.sqrt(np.mean((test["rating"].to_numpy() - train["rating"].mean()) ** 2)))


def score_matrices(model, train, user_ids, items):
    """User x movie score matrices of the five diagnostic systems; the last two do not depend on the user."""
    if not (np.array_equal(model.user_ids, user_ids) and np.array_equal(model.item_ids, items)):
        raise ValueError("model index differs from eg.build_index")
    bias = model.mu + model.bu[:, None] + model.bi[None, :]
    latent = model.P @ model.Q.T
    col = eg.locate(items, train["movieId"].to_numpy())
    count = np.bincount(col, minlength=len(items))
    total = np.bincount(col, weights=train["rating"].to_numpy(), minlength=len(items))
    shrunk = (total + PRIOR * train["rating"].mean()) / (count + PRIOR)
    full = np.clip(bias + latent, 1.0, 5.0)  # same operation order as MatrixFactorization.score_all
    mats = {"mf_full": full, "mf_bias_only": np.clip(bias, 1.0, 5.0), "mf_latent_only": latent,
            "shrunk_item_mean": np.broadcast_to(shrunk, full.shape), "item_bias_only": np.broadcast_to(model.bi, full.shape)}
    return mats, count


def check_against_file(full, path_scores, user_ids, items, seen, test, n_pairs, seed):
    """Max |model - mf_scores.csv| over random non-train pairs (the file is rounded to 4 decimals)."""
    file_mat, file_rmse = eg.load_mf(path_scores, user_ids, items, seen, test)
    rng = np.random.RandomState(seed)
    u = rng.randint(0, seen.shape[0], 2 * n_pairs)
    i = rng.randint(0, seen.shape[1], 2 * n_pairs)
    keep = ~seen[u, i]
    u, i = u[keep][:n_pairs], i[keep][:n_pairs]
    if len(u) != n_pairs:
        raise ValueError("could not draw %d non-train pairs" % n_pairs)
    return file_mat, float(np.abs(full[u, i] - file_mat[u, i]).max()), file_rmse


def group_contexts(groups, user_ids, seen, rel):
    """Per group: member rows, graded-member mask (>=1 liked test movie), candidate mask (unseen by all)."""
    n_liked = rel.sum(axis=1)
    out = []
    for members in groups:
        r = eg.locate(user_ids, members)
        out.append((r, n_liked[r] > 0, ~seen[r].any(axis=0)))
    return out, n_liked


def grade_system(scores, agg, ctx, rel, n_liked, k, allowed=None):
    """Top-k list per group (columns, ties by ascending movieId) and its metrics, as eval_group.evaluate_groups.

    scores: users x movies matrix aggregated over the members, or a 1-D vector used as the group score.
    allowed: optional movie mask ANDed into the candidate pool.
    """
    by_id = np.arange(scores.shape[-1])
    lists, rows = [], []
    for r, valid, cand in ctx:
        group_score = scores if scores.ndim == 1 else eg.AGGREGATORS[agg](scores[r])
        top, _ = eg.top_k(group_score, cand if allowed is None else cand & allowed, k, by_id)
        lists.append(top)
        if not valid.any():
            rows.append([np.nan] * 4)
            continue
        p, n = eg.member_metrics(rel[r[valid]], n_liked[r[valid]], top, k)
        rows.append([p.mean(), n.mean(), p.min(), n.min()])
    df = pd.DataFrame(rows, columns=GRADE_COLS)
    df.insert(0, "size", [len(c[0]) for c in ctx])
    return df, np.array(lists)


def sample_pairs(n_groups, n_pairs, seed):
    """n_pairs ordered pairs (a, b) of different groups, uniform, from RandomState(seed)."""
    rng = np.random.RandomState(seed)
    a = rng.randint(0, n_groups, n_pairs)
    return a, (a + rng.randint(1, n_groups, n_pairs)) % n_groups


def diversity(lists, items, titles, pairs):
    count = np.bincount(lists.ravel(), minlength=len(items))
    top = int(count.argmax())
    a, b = pairs
    overlap = (lists[a][:, :, None] == lists[b][:, None, :]).sum(axis=(1, 2))
    return {"distinct_movies": int((count > 0).sum()), "top_movie_share": count[top] / len(lists),
            "top_movie_title": titles.get(int(items[top]), str(items[top])), "mean_pair_overlap": overlap.mean()}


def system_rows(system, agg, df, div):
    rows = []
    for label in SIZE_LABELS:
        part = df if label == "all" else df[df["size"] == label]
        row = {"system": system, "agg": agg, "size": label, **part[GRADE_COLS].mean().to_dict()}
        rows.append({**row, **(div if label == "all" else {})})
    return rows


def existing_mf_average(path):
    """Per-group results of the team's mf / average / tie=movieId run, indexed by group_id."""
    df = pd.read_csv(path)
    df = df[(df["model"] == "mf") & (df["agg"] == "average") & (df["tie"] == "movieId")]
    return df.set_index("group_id").sort_index()


def validate_against_existing(name, df, existing, tol):
    diff = float(np.abs(df["p_mean"].to_numpy() - existing["p_mean"].to_numpy()).max())
    mean_diff = abs(df["p_mean"].mean() - existing["p_mean"].mean())
    print("  %-26s mean p_mean %.6f (existing %.6f), |diff| %.2e, max per-group |diff| %.2e"
          % (name, df["p_mean"].mean(), existing["p_mean"].mean(), mean_diff, diff))
    if mean_diff >= tol:
        raise ValueError("%s does not reproduce the existing MF / average result" % name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mf-scores", default=os.path.join(RESULTS, "mf_scores.csv"))
    ap.add_argument("--cache-dir", default=None, help="store/reuse fitted models (default: always retrain)")
    ap.add_argument("--out", default=os.path.join(OUT, "group_eval_diag_mf.csv"))
    ap.add_argument("--groups-per-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--check-pairs", type=int, default=200_000)
    ap.add_argument("--overlap-pairs", type=int, default=5000)
    args = ap.parse_args()
    t0 = time.time()

    print("[1/6] load split, masks and groups")
    train = eg.load_ratings(os.path.join(RESULTS, "ratings_train.csv"))
    test = eg.load_ratings(os.path.join(RESULTS, "ratings_test_eval.csv"))
    movies = pd.read_csv(os.path.join(RESULTS, "movies_processed.csv"), encoding="utf-8-sig", usecols=["movieId", "title"])
    titles = dict(zip(movies["movieId"], movies["title"]))
    user_ids, items, seen, rel = eg.build_index(train, test)
    groups = eg.sample_groups(user_ids, args.groups_per_size, args.seed)
    saved = pd.read_csv(os.path.join(OUT, "group_eval_groups.csv"))["userIds"]
    same_groups = all("|".join(map(str, g)) == s for g, s in zip(groups, saved)) and len(groups) == len(saved)
    print("  groups %d, identical to results/group_eval/group_eval_groups.csv: %s" % (len(groups), same_groups))
    ctx, n_liked = group_contexts(groups, user_ids, seen, rel)

    print("[2/6] retrain MF and the bias-only model, test RMSE")
    with open(os.path.join(RESULTS, "mf_best_params.json")) as f:
        params = json.load(f)
    model = fit_cached("mf_k%d" % params["k"], args.cache_dir, train, test, **params)
    rmse_mf = test_rmse(model, test)
    saved_rmse = float(pd.read_csv(os.path.join(RESULTS, "mf_final.csv"))["test_rmse"].iloc[0])
    print("  MF test RMSE %.10f (mf_final.csv %.10f, |diff| %.2e)" % (rmse_mf, saved_rmse, abs(rmse_mf - saved_rmse)))
    assert abs(rmse_mf - 0.8553) < 1e-3, "retrained MF does not reproduce the reported test RMSE"
    # k=0 gives zero-width P/Q, so MatrixFactorization itself runs the bias-only SGD (same update rule)
    bias_model = fit_cached("mf_k0", args.cache_dir, train, test, **{**params, "k": 0, "seed": 42})
    rmse_bias, rmse_global = test_rmse(bias_model, test), global_mean_rmse(train, test)
    print("  bias-only (k=0) test RMSE %.4f, global mean RMSE %.4f" % (rmse_bias, rmse_global))

    print("[3/6] score matrices and check against %s" % args.mf_scores)
    mats, count = score_matrices(model, train, user_ids, items)
    file_mat, max_diff, file_rmse = check_against_file(
        mats["mf_full"], args.mf_scores, user_ids, items, seen, test, args.check_pairs, args.seed)
    print("  %d random non-train pairs: max |model - file| = %.2e (file RMSE %.4f)" % (args.check_pairs, max_diff, file_rmse))
    assert max_diff < 1e-3, "retrained model does not reproduce mf_scores.csv"
    existing = existing_mf_average(os.path.join(OUT, "group_eval_by_group.csv")) if same_groups else None
    if existing is not None:  # the grading code below must give the team's per-group numbers from the same file
        file_graded, _ = grade_system(file_mat, "average", ctx, rel, n_liked, args.k)
        validate_against_existing("scores file, same code", file_graded, existing, 1e-9)
    del file_mat

    print("[4/6] grade systems")
    plan = [(name, agg, name, None) for name in ("mf_full", "mf_bias_only", "mf_latent_only")
            for agg in eg.AGGREGATORS]
    plan += [(name, "-", name, None) for name in ("shrunk_item_mean", "item_bias_only")]
    plan += [("mf_full_min%d" % n, "average", "mf_full", n) for n in MIN_COUNTS]
    graded = {}
    for system, agg, key, min_n in plan:
        allowed = None if min_n is None else count >= min_n
        graded[(system, agg)] = grade_system(
            mats[key][0] if agg == "-" else mats[key], agg, ctx, rel, n_liked, args.k, allowed)
        print("  %-18s %-12s p_mean %.4f ndcg_mean %.4f" % (
            system, agg, graded[(system, agg)][0]["p_mean"].mean(), graded[(system, agg)][0]["ndcg_mean"].mean()))

    print("[5/6] validate against the existing evaluation")
    if existing is not None:
        df = graded[("mf_full", "average")][0]
        diff = abs(df["p_mean"].mean() - existing["p_mean"].mean())
        print("  mf_full / average vs existing: |diff| %.2e (%s)" % (diff, "exact" if diff < 1e-9 else "4-decimal rounding of the file"))
        assert diff < 5e-4
        rounded, _ = grade_system(np.round(mats["mf_full"], 4), "average", ctx, rel, n_liked, args.k)
        validate_against_existing("model rounded to 4 dec.", rounded, existing, 1e-9)
    else:
        print("  skipped: groups differ from results/group_eval/group_eval_groups.csv")

    print("[6/6] diversity, save")
    pairs = sample_pairs(len(groups), args.overlap_pairs, args.seed)
    rows = []
    for (system, agg), (df, lists) in graded.items():
        rows += system_rows(system, agg, df, diversity(lists, items, titles, pairs))
    for agg, value in (("mf_full", rmse_mf), ("bias_only_k0", rmse_bias), ("global_mean", rmse_global)):
        rows.append({"system": "rmse_check", "agg": agg, "size": "all", "rmse": value})
    out = pd.DataFrame(rows, columns=COLUMNS)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)

    pd.set_option("display.width", 220)
    overall = out[out["size"] == "all"].drop(columns=["size", "p_min", "ndcg_min"])
    print(overall.round(4).to_string(index=False))
    for agg in eg.AGGREGATORS:
        same = (graded[("mf_bias_only", agg)][1] == graded[("item_bias_only", "-")][1]).all(axis=1).mean()
        print("  share of groups with mf_bias_only (%s) list == item_bias_only list: %.3f" % (agg, same))
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
