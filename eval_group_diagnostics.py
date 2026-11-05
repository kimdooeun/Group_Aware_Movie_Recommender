"""Diagnostics behind the caveats in results/group_eval/group_eval_report.md.

    python eval_group_diagnostics.py --mf-scores /path/to/mf_scores.csv

Needs the results/group_eval/group_eval_*.csv written by eval_group.py and eval_group_report.py. Adds (results/group_eval/):
    group_eval_diag_lists.csv       how many distinct movies the group lists contain, overlap between lists
    group_eval_diag_popmatched.csv  MF vs content-based when candidates are limited to movies with >= N train ratings
    group_eval_diag_cohesion.csv    random / popularity / model levels per cohesion tercile (lift over random)
    group_eval_diag_seeds.csv       the headline numbers for groups drawn with other seeds
    group_eval_diag_meta.json       how often users are reused across groups
The MF bias / latent decomposition is in eval_group_diag_mf.py.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import eval_group as eg

RESULTS = eg.RESULTS
OUT = eg.OUT_DIR  # group_eval_* files live here
K = 10
MIN_RATINGS = (0, 50, 100, 200)
SEEDS = (42, 1, 2, 3)


def run(groups, systems, user_ids, seen, rel, pool=None, k=K):
    """Top-k list per group and system plus P@k / NDCG@k (member mean), graded like eval_group.evaluate_groups."""
    n_liked = rel.sum(axis=1)
    by_id = np.arange(seen.shape[1])
    lists = {s: np.zeros((len(groups), k), dtype=np.int64) for s in systems}
    metrics = {s: np.full((len(groups), 2), np.nan) for s in systems}
    random_p, n_cand = np.zeros(len(groups)), np.zeros(len(groups))
    for gid, members in enumerate(groups):
        r_all = eg.locate(user_ids, members)
        r = r_all[n_liked[r_all] > 0]
        cand = ~seen[r_all].any(axis=0)
        if pool is not None:
            cand &= pool
        n_cand[gid] = cand.sum()
        random_p[gid] = ((rel[r] & cand).sum(axis=1) / cand.sum()).mean()
        for name, fn in systems.items():
            top, _ = eg.top_k(fn(r_all), cand, k, by_id)
            lists[name][gid] = top
            p, n = eg.member_metrics(rel[r], n_liked[r], top, k)
            metrics[name][gid] = (p.mean(), n.mean())
    return lists, metrics, random_p, n_cand


def diversity(lists, titles, rng, n_pairs=5000):
    rows = []
    for name, arr in lists.items():
        counts = np.bincount(arr.ravel())
        i = rng.randint(0, len(arr), n_pairs)
        j = (i + rng.randint(1, len(arr), n_pairs)) % len(arr)
        overlap = np.mean([len(set(arr[a]) & set(arr[b])) for a, b in zip(i, j)])
        rows.append({"system": name, "distinct_movies": int((counts > 0).sum()),
                     "top_movie": titles[counts.argmax()], "top_movie_share": counts.max() / len(arr),
                     "mean_overlap_random_pairs": overlap})
    out = pd.DataFrame(rows)
    for model in eg.MODELS:  # does the aggregation rule change the list at all?
        a, b = lists[model + "_average"], lists[model + "_least_misery"]
        out.loc[out["system"] == model + "_average", "average_vs_least_misery_overlap"] = \
            np.mean([len(set(x) & set(y)) for x, y in zip(a, b)])
    return out


def popularity_matched(groups, scores, pop, user_ids, seen, rel):
    systems = {"mf": lambda r: eg.AGGREGATORS["average"](scores["mf"][r]),
               "content": lambda r: eg.AGGREGATORS["average"](scores["content"][r]), "popularity": lambda r: pop}
    rows = []
    for n_min in MIN_RATINGS:
        _, metrics, random_p, n_cand = run(groups, systems, user_ids, seen, rel, pool=pop >= n_min)
        levels = {"random": (random_p.mean(), np.nan), **{s: tuple(np.nanmean(m, axis=0)) for s, m in metrics.items()}}
        for name, (p, n) in levels.items():
            rows.append({"min_train_ratings": n_min, "candidates_mean": n_cand.mean(), "system": name, "p_mean": p,
                         "ndcg_mean": n, "lift_over_random": p / random_p.mean()})
    return pd.DataFrame(rows)


def cohesion_baselines():
    coh = pd.read_csv(os.path.join(OUT, "group_eval_cohesion.csv")).dropna(subset=["tercile"])
    base = pd.read_csv(os.path.join(OUT, "group_eval_baselines.csv"))[["group_id", "random_p_mean", "popularity_p_mean"]]
    bg = pd.read_csv(os.path.join(OUT, "group_eval_by_group.csv"))
    bg = bg[(bg["tie"] == "movieId") & (bg["agg"] == "average")].pivot(index="group_id", columns="model", values="p_mean")
    df = coh.merge(base, on="group_id").merge(bg.reset_index(), on="group_id")
    rows = []
    for t in ("Low", "Mid", "High"):
        sub = df[df["tercile"] == t]
        rand = sub["random_p_mean"].mean()
        for name, col in (("random", "random_p_mean"), ("popularity", "popularity_p_mean"), ("mf", "mf"),
                          ("content", "content")):
            rows.append({"tercile": t, "system": name, "p_mean": sub[col].mean(), "lift_over_random": sub[col].mean() / rand,
                         "n_groups": len(sub)})
    return pd.DataFrame(rows)


def seed_repeats(scores, pop, user_ids, seen, rel):
    systems = {"popularity": lambda r: pop}
    for model in eg.MODELS:
        for agg, fn in eg.AGGREGATORS.items():
            systems["%s_%s" % (model, agg)] = lambda r, m=model, f=fn: f(scores[m][r])
    rows = []
    for seed in SEEDS:
        groups = eg.sample_groups(user_ids, 1000, seed)
        _, metrics, random_p, _ = run(groups, systems, user_ids, seen, rel)
        rows.append({"seed": seed, "system": "random", "p_mean": random_p.mean(), "ndcg_mean": np.nan})
        rows += [{"seed": seed, "system": s, "p_mean": m[:, 0].mean(), "ndcg_mean": m[:, 1].mean()}
                 for s, m in metrics.items()]
        print("  seed %d done" % seed)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mf-scores", default=os.path.join(RESULTS, "mf_scores.csv"))
    args = ap.parse_args()
    rng = np.random.RandomState(42)
    with open(os.path.join(OUT, "group_eval_meta.json")) as f:
        meta = json.load(f)

    print("[1/5] load split, groups and scores")
    train = eg.load_ratings(os.path.join(RESULTS, "ratings_train.csv"))
    test = eg.load_ratings(os.path.join(RESULTS, "ratings_test_eval.csv"))
    movies = pd.read_csv(os.path.join(RESULTS, "movies_processed.csv"), encoding="utf-8-sig")
    user_ids, items, seen, rel = eg.build_index(train, test)
    gdf = pd.read_csv(os.path.join(OUT, "group_eval_groups.csv"))
    groups = [np.array(s.split("|"), dtype=np.int64) for s in gdf["userIds"]]
    assert all(np.array_equal(a, b) for a, b in zip(groups, eg.sample_groups(user_ids, meta["groups_per_size"], meta["seed"])))
    mf, _ = eg.load_mf(args.mf_scores, user_ids, items, seen, test)
    scores = {"mf": mf, "content": eg.load_content(user_ids, user_ids, items)}  # every user: other seeds need them
    pop = np.bincount(eg.locate(items, train["movieId"].to_numpy()), minlength=len(items)).astype(float)
    titles = dict(zip(movies["movieId"], movies["title"]))
    col_title = {i: titles[m] for i, m in enumerate(items)}

    print("[2/5] list diversity")
    systems = {"popularity": lambda r: pop}
    for model in eg.MODELS:
        for agg, fn in eg.AGGREGATORS.items():
            systems["%s_%s" % (model, agg)] = lambda r, m=model, f=fn: f(scores[m][r])
    lists, metrics, _, _ = run(groups, systems, user_ids, seen, rel)
    summary = pd.read_csv(os.path.join(OUT, "group_eval_summary.csv"))
    ref = summary[(summary["size"] == "all") & (summary["tie"] == "movieId")].set_index(["model", "agg"])["p_mean"]
    for model in eg.MODELS:
        for agg in eg.AGGREGATORS:
            got = metrics["%s_%s" % (model, agg)][:, 0].mean()
            assert abs(got - ref[(model, agg)]) < 1e-9, (model, agg, got, ref[(model, agg)])
    div = diversity(lists, col_title, rng)
    div.to_csv(os.path.join(OUT, "group_eval_diag_lists.csv"), index=False)
    print(div.round(3).to_string(index=False))

    print("[3/5] popularity-matched comparison")
    pm = popularity_matched(groups, scores, pop, user_ids, seen, rel)
    pm.to_csv(os.path.join(OUT, "group_eval_diag_popmatched.csv"), index=False)
    print(pm.round(4).to_string(index=False))

    print("[4/5] cohesion baselines")
    cb = cohesion_baselines()
    cb.to_csv(os.path.join(OUT, "group_eval_diag_cohesion.csv"), index=False)
    print(cb.round(4).to_string(index=False))

    print("[5/5] other seeds and user reuse")
    seeds = seed_repeats(scores, pop, user_ids, seen, rel)
    seeds.to_csv(os.path.join(OUT, "group_eval_diag_seeds.csv"), index=False)
    print(seeds.groupby("system")[["p_mean", "ndcg_mean"]].agg(["mean", "std"]).round(4).to_string())
    uses = np.bincount(np.concatenate(groups))
    uses = uses[uses > 0]
    reuse = {"member_slots": int(uses.sum()), "distinct_users": int(len(uses)),
             "share_in_two_or_more_groups": float((uses >= 2).mean()), "max_groups_per_user": int(uses.max())}
    with open(os.path.join(OUT, "group_eval_diag_meta.json"), "w") as f:
        json.dump(reuse, f, indent=2)
    print(reuse)


if __name__ == "__main__":
    main()
