"""Reference lines, cohesion split, distribution plots and case studies for the group evaluation.

    python eval_group_report.py --mf-scores /path/to/mf_scores.csv

Needs the results/group_eval/group_eval_*.csv written by eval_group.py (same groups, same K, same tie rule: movieId).
Adds (results/group_eval/): group_eval_individual.csv, group_eval_individual_summary.csv, group_eval_cohesion.csv,
group_eval_cohesion_summary.csv, group_eval_cohesion_lm_vs_avg.csv and results/group_eval/figures/*.png.

Individual reference: every member gets the Top-K of their own scores (candidates = movies they have not
rated in train), graded exactly like the group list. Cohesion: mean pairwise cosine similarity of the members'
genre vectors (genre counts of train movies rated >= 4); groups are split into terciles within each size.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eval_group as eg
from mf import load_ratings

RESULTS = eg.RESULTS
OUT = eg.OUT_DIR  # group_eval_* files live here
FIGURES = os.path.join(OUT, "figures")
AGGS = ("average", "least_misery")
AGG_LABEL = {"average": "Average", "least_misery": "Least Misery"}
MODEL_LABEL = {"mf": "MF", "content": "Content-based"}
TERCILES = ("Low", "Mid", "High")

# validated reference palette (light surface): MF = categorical slot 1, content-based = slot 2
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"mf": "#2a78d6", "content": "#eb6834"}
REF = MUTED


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "font.size": 10.5, "axes.titlesize": 11.5,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "legend.frameon": False,
    })


def header(fig, title, subtitle=None):
    """Title block pinned to the top in inches, so every figure has the same spacing."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, title, ha="left", va="top", fontsize=14, fontweight="semibold", color=INK)
    if subtitle:
        fig.text(0.01, 1 - 0.55 / h, subtitle, ha="left", va="top", fontsize=10.5, color=INK2)


def save(fig, name, title, subtitle=None, legend=None):
    h = fig.get_figheight()
    bottom = 0.0
    if legend:
        fig.legend(*legend, loc="lower center", ncol=len(legend[1]), bbox_to_anchor=(0.5, 0.0))
        bottom = 0.55
    fig.tight_layout(rect=(0, bottom / h, 1, 1 - 1.0 / h))
    header(fig, title, subtitle)
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved", path)


def boot_mean_ci(x, rng, n_boot=2000):
    x = np.asarray(x, dtype=float)
    boot = x[rng.randint(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return x.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def with_all(df):
    return pd.concat([df.assign(size=df["size"].astype(str)), df.assign(size="All")])


# ---------------------------------------------------------------------------------------------- analysis

def individual_reference(groups, scores, user_ids, seen, rel, k):
    """Per-group metrics when every member gets their own Top-K (no aggregation)."""
    n_liked = rel.sum(axis=1)
    by_id = np.arange(seen.shape[1])
    needed = np.unique(np.concatenate(groups))
    rows_needed = eg.locate(user_ids, needed)
    per_user = {}
    for model in eg.MODELS:
        p, n = np.full(len(user_ids), np.nan), np.full(len(user_ids), np.nan)
        for r in rows_needed:
            if n_liked[r] == 0:
                continue
            top, _ = eg.top_k(scores[model][r], ~seen[r], k, by_id)
            pp, nn = eg.member_metrics(rel[r:r + 1], n_liked[r:r + 1], top, k)
            p[r], n[r] = pp[0], nn[0]
        per_user[model] = (p, n)
    rows = []
    for gid, members in enumerate(groups):
        r = eg.locate(user_ids, members)
        r = r[n_liked[r] > 0]
        for model in eg.MODELS:
            p, n = per_user[model][0][r], per_user[model][1][r]
            rows.append({"group_id": gid, "size": len(members), "model": model, "p_mean": p.mean(), "p_min": p.min(),
                         "ndcg_mean": n.mean(), "ndcg_min": n.min()})
    return pd.DataFrame(rows)


def individual_same_pool(groups, scores, user_ids, seen, rel, k):
    """Individual reference drawn from the group's own candidate pool, so only aggregation differs.

    Also returns, per group size, the share of graded members' liked test movies that the group pool excludes
    (another member already rated them in train, so no group list can contain them)."""
    n_liked = rel.sum(axis=1)
    by_id = np.arange(seen.shape[1])
    rows, lost = [], []
    for gid, members in enumerate(groups):
        r_all = eg.locate(user_ids, members)
        r = r_all[n_liked[r_all] > 0]
        cand = ~seen[r_all].any(axis=0)
        lost.append({"size": len(members), "excluded": int((rel[r] & ~cand).sum()), "liked": int(rel[r].sum())})
        for model in eg.MODELS:
            p, n = [], []
            for u in r:
                top, _ = eg.top_k(scores[model][u], cand, k, by_id)
                pp, nn = eg.member_metrics(rel[u:u + 1], n_liked[u:u + 1], top, k)
                p.append(pp[0])
                n.append(nn[0])
            rows.append({"group_id": gid, "size": len(members), "model": model, "p_mean": np.mean(p), "p_min": np.min(p),
                         "ndcg_mean": np.mean(n), "ndcg_min": np.min(n)})
    loss = pd.DataFrame(lost).groupby("size")[["excluded", "liked"]].sum()
    loss.loc["All"] = loss.sum()
    loss["excluded_share"] = loss["excluded"] / loss["liked"]
    return pd.DataFrame(rows), loss.reset_index()


def genre_vectors(movies, items, train, user_ids):
    """Users x genres counts of train movies rated >= 4 (rows follow user_ids)."""
    genres = sorted({g for s in movies["genres"] for g in s.split("|")})
    gmat = np.zeros((len(items), len(genres)))
    col = {g: i for i, g in enumerate(genres)}
    row_of = dict(zip(movies["movieId"], movies["genres"]))
    for i, mid in enumerate(items):
        for g in row_of[mid].split("|"):
            gmat[i, col[g]] = 1
    liked = train[train["rating"] >= 4]
    lmat = np.zeros((len(user_ids), len(items)))
    lmat[eg.locate(user_ids, liked["userId"].to_numpy()), eg.locate(items, liked["movieId"].to_numpy())] = 1
    return lmat @ gmat, genres


def cohesion(groups, vectors, user_ids):
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norm, out=np.full_like(vectors, np.nan), where=norm > 0)
    out = []
    for members in groups:
        u = unit[eg.locate(user_ids, members)]
        sim = u @ u.T
        pairs = sim[np.triu_indices(len(members), 1)]
        out.append(np.nan if np.isnan(pairs).all() else np.nanmean(pairs))
    return np.array(out)


def tercile_labels(coh, sizes):
    label = np.full(len(coh), None, dtype=object)
    for size in np.unique(sizes):
        m = (sizes == size) & ~np.isnan(coh)
        label[m] = pd.qcut(coh[m], 3, labels=TERCILES).astype(str)
    return label


# ---------------------------------------------------------------------------------------------- figures

def style_axes(ax, ylabel=None, xlabel=None):
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.tick_params(axis="both", labelsize=10)


def fig_context(ind, bg, k, ind2):
    """Group list vs the same members' own Top-K lists (ind: own pool, ind2: the group's pool)."""
    labels = ["2", "3", "4", "5", "All"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharey=True)
    bg, ind, ind2 = with_all(bg), with_all(ind), with_all(ind2)
    rows = []
    for r, agg in enumerate(AGGS):
        for c, model in enumerate(eg.MODELS):
            ax = axes[r, c]
            g = bg[(bg["model"] == model) & (bg["agg"] == agg) & (bg["tie"] == "movieId")].groupby("size")["p_mean"].mean()
            i = ind[ind["model"] == model].groupby("size")["p_mean"].mean()
            i2 = ind2[ind2["model"] == model].groupby("size")["p_mean"].mean()
            x, w, gap = np.arange(5), 0.24, 0.015
            ax.bar(x - w - 2 * gap, [i[s] for s in labels], w, color=REF, zorder=3)
            ax.bar(x, [i2[s] for s in labels], w, color=AXIS, zorder=3)
            ax.bar(x + w + 2 * gap, [g[s] for s in labels], w, color=COLOR[model], zorder=3)
            ax.text(4, i["All"] + 0.0025, "group list =\n%.0f%% / %.0f%% of individual\n(own / group's pool)"
                    % (g["All"] / i["All"] * 100, g["All"] / i2["All"] * 100), ha="center", va="bottom", fontsize=9,
                    color=INK2)
            ax.set_ylim(0, 0.102)
            ax.set_xticks(x, labels)
            ax.set_title("%s · %s" % (MODEL_LABEL[model], AGG_LABEL[agg]))
            style_axes(ax, "Precision@%d (mean over members)" % k if c == 0 else None,
                       "Group size" if r == 1 else None)
            rows += [{"model": model, "agg": agg, "size": s, "individual_p_mean": i[s], "group_p_mean": g[s],
                      "group_over_individual": g[s] / i[s], "individual_samepool_p_mean": i2[s],
                      "group_over_individual_samepool": g[s] / i2[s]} for s in labels]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (REF, AXIS, COLOR["mf"], COLOR["content"])]
    save(fig, "fig1_group_vs_individual.png", "One shared list keeps only part of what personal lists reach",
         "Mean precision over members, 4,000 random groups. Gray = the same members' own Top-%d lists,\n"
         "picked from any movie they have not rated (dark) or only from the group's pool (light)." % k,
         legend=(handles, ["Individual · own pool", "Individual · group's pool", "Group list · MF",
                           "Group list · Content-based"]))
    return pd.DataFrame(rows)


def fig_distribution(bg, k):
    sub = bg[bg["tie"] == "movieId"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharey="col")
    sizes = ["2", "3", "4", "5", "All"]
    buckets = [("0 (no hits)", lambda v: v == 0), ("(0, 0.05]", lambda v: (v > 0) & (v <= 0.05)),
               ("(0.05, 0.10]", lambda v: (v > 0.05) & (v <= 0.10)), ("> 0.10", lambda v: v > 0.10)]
    for r, agg in enumerate(AGGS):
        s = with_all(sub[sub["agg"] == agg])
        ax = axes[r, 0]
        x, w, gap = np.arange(5), 0.30, 0.02
        for j, model in enumerate(eg.MODELS):
            share = [100 * (s[(s["model"] == model) & (s["size"] == z)]["p_mean"] > 0).mean() for z in sizes]
            ax.bar(x + (j - 0.5) * (w + 2 * gap), share, w, color=COLOR[model], zorder=3)
            ax.text(4 + (j - 0.5) * (w + 2 * gap), share[-1] + 1.5, "%.0f%%" % share[-1], ha="center", fontsize=9.5,
                    color=INK2)
        ax.set_xticks(x, sizes)
        ax.set_ylim(0, 100)
        ax.set_title("%s: groups where any member has a hit" % AGG_LABEL[agg])
        style_axes(ax, "% of groups", "Group size" if r == 1 else None)

        ax = axes[r, 1]
        pooled = sub[sub["agg"] == agg]
        x = np.arange(len(buckets))
        for j, model in enumerate(eg.MODELS):
            v = np.round(pooled[pooled["model"] == model]["p_mean"].to_numpy(), 9)  # keep 0.1 on its side of the edge
            share = [100 * f(v).mean() for _, f in buckets]
            ax.bar(x + (j - 0.5) * (w + 2 * gap), share, w, color=COLOR[model], zorder=3)
            ax.text(0 + (j - 0.5) * (w + 2 * gap), share[0] + 1.5, "%.0f%%" % share[0], ha="center", fontsize=9.5,
                    color=INK2)
        ax.set_xticks(x, [b[0] for b in buckets])
        ax.set_ylim(0, 100)
        ax.set_title("%s: spread of the group's P@%d" % (AGG_LABEL[agg], k))
        style_axes(ax, None, "Group P@%d (mean over members)" % k if r == 1 else None)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in eg.MODELS]
    save(fig, "fig2_distribution.png", "Most content-based lists miss entirely; MF lists usually hit something",
         "A hit = a member liked that movie in the test split. Left panels count a group once any member hits, so they "
         "rise with group size while per-member precision falls. 4,000 groups.",
         legend=(handles, [MODEL_LABEL[m] for m in eg.MODELS]))


def fig_paired(bg, k):
    sub = bg[bg["tie"] == "movieId"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), gridspec_kw={"width_ratios": [1.15, 1]})
    sizes = ["2", "3", "4", "5", "All"]
    edges = np.arange(-0.41, 0.42, 0.02)
    out = []
    for r, agg in enumerate(AGGS):
        w = with_all(sub[sub["agg"] == agg]).pivot_table(index=["group_id", "size"], columns="model", values="p_mean")
        d = (w["mf"] - w["content"]).round(9).reset_index(name="d")  # equal hit counts can differ by ~1e-16
        ax = axes[r, 0]
        dd = d[d["size"] == "All"]["d"].to_numpy()
        counts, _ = np.histogram(dd, bins=edges)
        assert counts.sum() == len(dd), "histogram range drops groups"
        centers = (edges[:-1] + edges[1:]) / 2
        colors = [COLOR["mf"] if c > 0.0051 else COLOR["content"] if c < -0.0051 else REF for c in centers]
        ax.bar(centers, counts, width=0.018, color=colors, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 8000)
        ax.set_xlim(-0.43, 0.43)
        ax.set_title("%s: MF minus content-based, per group" % AGG_LABEL[agg])
        style_axes(ax, "Groups (log scale)", "Difference in group P@%d" % k if r == 1 else None)
        ax.text(0.41, 3000, "MF better", ha="right", color=INK2, fontsize=10)
        ax.text(-0.41, 3000, "Content-based better", ha="left", color=INK2, fontsize=10)

        ax = axes[r, 1]
        for yi, z in enumerate(reversed(sizes)):
            x = d[d["size"] == z]["d"].to_numpy()
            parts = [(x > 0).mean() * 100, (x == 0).mean() * 100, (x < 0).mean() * 100]
            out.append({"agg": agg, "size": z, "mf_better_pct": parts[0], "tie_pct": parts[1],
                        "content_better_pct": parts[2]})
            left = 0.0
            for val, colr, ink in zip(parts, (COLOR["mf"], AXIS, COLOR["content"]), ("#ffffff", INK, "#ffffff")):
                seg = max(val - 0.4, 0)
                ax.barh(yi, seg, left=left, height=0.46, color=colr, zorder=3)
                if val >= 5:
                    ax.text(left + seg / 2, yi, "%.0f%%" % val, ha="center", va="center", fontsize=9.5, color=ink)
                left += val
        ax.set_yticks(range(len(sizes)), [("Size %s" % z if z != "All" else "All") for z in reversed(sizes)])
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_title("%s: win / tie / loss" % AGG_LABEL[agg])
        style_axes(ax, None, "% of groups" if r == 1 else None)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (COLOR["mf"], AXIS, COLOR["content"])]
    save(fig, "fig3_paired_difference.png", "Same groups, two models: MF wins far more often than it loses",
         "Per-group P@%d difference. Groups are identical for both models, so the comparison is paired." % k,
         legend=(handles, ["MF better", "Tie (mostly both zero)", "Content-based better"]))
    return pd.DataFrame(out)


def fig_forest(diff):
    d = diff[diff["tie"] == "movieId"]
    metrics = [("p_mean", "P@10, member mean"), ("ndcg_mean", "NDCG@10, member mean"),
               ("p_min", "P@10, member min"), ("ndcg_min", "NDCG@10, member min")]
    order = ["all", "2", "3", "4", "5"]  # eval_group.py labels the pooled row "all"
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.4), sharey=True)
    style = {"average": ("o", INK), "least_misery": ("D", MUTED)}
    for ax, (m, title) in zip(axes, metrics):
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.axvline(0, color=AXIS, linewidth=1.2, zorder=1)
        for j, agg in enumerate(AGGS):
            s = d[(d["metric"] == m) & (d["agg"] == agg)].set_index("size")
            for yi, z in enumerate(order):
                y = yi + (j - 0.5) * 0.22
                row = s.loc[z]
                ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=style[agg][1], linewidth=2, zorder=2,
                        solid_capstyle="round")
                ax.plot(row["mf_minus_content"], y, style[agg][0], color=style[agg][1], markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.set_yticks(range(len(order)), [("Size %s" % z if z != "all" else "All sizes") for z in order])
        ax.set_title(title)
        style_axes(ax, None, "MF − content-based")
    axes[0].invert_yaxis()  # shared y: invert once so "All sizes" is on top
    handles = [plt.Line2D([], [], marker=style[a][0], color=style[a][1], linewidth=2, markersize=7,
                          markeredgecolor=SURFACE) for a in AGGS]
    save(fig, "fig4_mf_minus_content_ci.png",
         "MF leads on the member mean; the member minimum cannot separate the models in larger groups",
         "Paired bootstrap, 95% CI, 2,000 resamples over groups. Where an interval touches 0 the metric cannot "
         "tell the models apart.", legend=(handles, [AGG_LABEL[a] for a in AGGS]))


def fig_cohesion(coh_df, k, rng):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharey="row")
    rows = []
    for r, (metric, ylab) in enumerate((("p_mean", "Precision@%d" % k), ("ndcg_mean", "NDCG@%d" % k))):
        for c, agg in enumerate(AGGS):
            ax = axes[r, c]
            x, w, gap = np.arange(3), 0.30, 0.02
            for j, model in enumerate(eg.MODELS):
                stats = []
                for t in TERCILES:
                    v = coh_df[(coh_df["model"] == model) & (coh_df["agg"] == agg) & (coh_df["tercile"] == t)][metric]
                    stats.append(boot_mean_ci(v, rng))
                    rows.append({"metric": metric, "model": model, "agg": agg, "tercile": t, "n_groups": len(v),
                                 "mean": stats[-1][0], "ci_low": stats[-1][1], "ci_high": stats[-1][2]})
                pos = x + (j - 0.5) * (w + 2 * gap)
                m = np.array([s[0] for s in stats])
                ax.bar(pos, m, w, color=COLOR[model], zorder=3)
                ax.errorbar(pos, m, yerr=[m - [s[1] for s in stats], [s[2] for s in stats] - m], fmt="none",
                            ecolor=INK2, elinewidth=1.2, capsize=0, zorder=4)
            ax.set_xticks(x, ["Low", "Mid", "High"])
            ax.set_title("%s: %s" % (AGG_LABEL[agg], ylab))
            style_axes(ax, "%s (mean over members)" % ylab if c == 0 else None,
                       "Group taste similarity (genre-vector cosine)" if r == 1 else None)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in eg.MODELS]
    save(fig, "fig5_cohesion.png", "Performance by how alike the group members are",
         "Terciles are cut within each group size, so size does not mix in. Bars = mean with 95% bootstrap CI.",
         legend=(handles, [MODEL_LABEL[m] for m in eg.MODELS]))
    return pd.DataFrame(rows)


def fig_cohesion_lift(coh_df, base, k):
    """Cohesive groups simply have more liked movies; the lift over a random list shows what the model adds."""
    rng = np.random.RandomState(7)  # separate stream: leaves the other bootstrap results unchanged
    d = coh_df.merge(base[["group_id", "random_p_mean"]], on="group_id")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    rows = []
    for ax, agg in zip(axes, AGGS):
        x, w, gap = np.arange(3), 0.30, 0.02
        for j, model in enumerate(eg.MODELS):
            vals = []
            for t in TERCILES:
                s = d[(d["model"] == model) & (d["agg"] == agg) & (d["tercile"] == t)]
                m, rr = s["p_mean"].to_numpy(), s["random_p_mean"].to_numpy()
                pick = rng.randint(0, len(m), (2000, len(m)))
                boot = m[pick].mean(axis=1) / rr[pick].mean(axis=1)
                vals.append((m.mean() / rr.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)))
                rows.append({"model": model, "agg": agg, "tercile": t, "random_p_mean": rr.mean(), "p_mean": m.mean(),
                             "lift": vals[-1][0], "ci_low": vals[-1][1], "ci_high": vals[-1][2]})
            pos = x + (j - 0.5) * (w + 2 * gap)
            lift = np.array([v[0] for v in vals])
            ax.bar(pos, lift, w, color=COLOR[model], zorder=3)
            ax.errorbar(pos, lift, yerr=[lift - [v[1] for v in vals], [v[2] for v in vals] - lift], fmt="none",
                        ecolor=INK2, elinewidth=1.2, capsize=0, zorder=4)
        ax.set_xticks(x, ["Low", "Mid", "High"])
        ax.set_title("%s: P@%d divided by random P@%d" % (AGG_LABEL[agg], k, k))
        style_axes(ax, "Lift over a random list" if agg == AGGS[0] else None, "Group taste similarity (genre-vector cosine)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in eg.MODELS]
    save(fig, "fig5b_cohesion_lift.png", "Similar groups score higher because they have more liked movies, not because MF does better",
         "A random list also scores higher in similar groups (more liked movies to hit), so lift over random is the fair "
         "comparison. Bars = ratio of means with 95% bootstrap CI.", legend=(handles, [MODEL_LABEL[m] for m in eg.MODELS]))
    return pd.DataFrame(rows)


def fig_lm_vs_avg(coh_df, rng, bg):
    d = coh_df.pivot_table(index=["group_id", "model", "tercile"], columns="agg", values=["p_mean", "p_min"])
    pooled = bg[bg["tie"] == "movieId"].pivot_table(index=["group_id", "model"], columns="agg", values=["p_mean", "p_min"])
    rng_all = np.random.RandomState(11)  # separate stream: leaves the tercile bootstraps unchanged
    rows = list(TERCILES) + ["All groups"]
    out = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=False)
    for ax, (m, title) in zip(axes, (("p_mean", "P@10, member mean"), ("p_min", "P@10, member min"))):
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.axvline(0, color=AXIS, linewidth=1.2, zorder=1)
        for j, model in enumerate(eg.MODELS):
            for yi, t in enumerate(rows):
                if t == "All groups":
                    x = (pooled[m]["least_misery"] - pooled[m]["average"]).xs(model, level="model").round(9).to_numpy()
                    mean, lo, hi = boot_mean_ci(x, rng_all)
                else:
                    x = (d[m]["least_misery"] - d[m]["average"]).xs(model, level="model").xs(t, level="tercile").to_numpy()
                    mean, lo, hi = boot_mean_ci(x, rng)
                out.append({"metric": m, "model": model, "tercile": t, "n_groups": len(x), "lm_minus_avg": mean,
                            "ci_low": lo, "ci_high": hi})
                y = yi + (j - 0.5) * 0.24
                ax.plot([lo, hi], [y, y], color=COLOR[model], linewidth=2, zorder=2, solid_capstyle="round")
                ax.plot(mean, y, "o", color=COLOR[model], markersize=7, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, zorder=3)
        ax.set_yticks(range(len(rows)), ["%s similarity" % t if t in TERCILES else t for t in rows])
        ax.invert_yaxis()
        ax.set_title(title)
        style_axes(ax, None, "Least Misery − Average")
    handles = [plt.Line2D([], [], marker="o", color=COLOR[m], linewidth=2, markersize=7, markeredgecolor=SURFACE)
               for m in eg.MODELS]
    save(fig, "fig6_least_misery_vs_average.png",
         "Least Misery gives no gain over Average in P@10 (the member minimum is mostly 0)",
         "Paired difference per group, 95% bootstrap CI; right of 0 = Least Misery better. Mind the axis: effects "
         "are within about 0.003 P@10, against levels of 0.03 (MF) and 0.008 (content-based).",
         legend=(handles, [MODEL_LABEL[m] for m in eg.MODELS]))
    return pd.DataFrame(out)


def reference_levels(groups, scores, user_ids, items, seen, rel, train, k):
    """Reference points for the same groups, graded exactly like the models.

    random: exact expectation (a random list holds a liked movie in each slot with probability |liked & pool| / |pool|);
    popularity: movies ranked by number of train ratings (ties: ascending movieId), no personalisation;
    upper bound: the best possible shared list (the k candidates liked by most graded members) / best personal list.
    Also the median popularity (train rating count) of what each kind of list contains."""
    n_liked = rel.sum(axis=1)
    pop = np.bincount(eg.locate(items, train["movieId"].to_numpy()), minlength=len(items)).astype(float)
    by_id = np.arange(len(items))
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = np.concatenate([[0.0], np.cumsum(discount)])
    pop_own = {}
    rows, contents = [], []
    for gid, members in enumerate(groups):
        r_all = eg.locate(user_ids, members)
        r = r_all[n_liked[r_all] > 0]
        cand = ~seen[r_all].any(axis=0)
        q = (rel[r] & cand).sum(axis=1) / cand.sum()
        top, _ = eg.top_k(pop, cand, k, by_id)
        p, n = eg.member_metrics(rel[r], n_liked[r], top, k)
        ip, inn = [], []
        for u in r:
            if u not in pop_own:
                t, _ = eg.top_k(pop, ~seen[u], k, by_id)
                pop_own[u] = tuple(v[0] for v in eg.member_metrics(rel[u:u + 1], n_liked[u:u + 1], t, k))
            ip.append(pop_own[u][0])
            inn.append(pop_own[u][1])
        counts = np.where(cand, rel[r].sum(axis=0), -1)
        rows.append({"group_id": gid, "size": len(members),
                     "random_p_mean": q.mean(), "random_ndcg_mean": (q * discount.sum() / idcg[np.minimum(n_liked[r], k)]).mean(),
                     "popularity_p_mean": p.mean(), "popularity_p_min": p.min(),
                     "popularity_ndcg_mean": n.mean(), "popularity_ndcg_min": n.min(),
                     "popularity_individual_p_mean": np.mean(ip), "popularity_individual_ndcg_mean": np.mean(inn),
                     "upper_bound_group_p_mean": np.sort(counts)[-k:].sum() / (k * len(r)),
                     "upper_bound_individual_p_mean": np.mean(np.minimum(n_liked[r], k) / k)})
        liked_items = np.where((rel[r] & cand).any(axis=0))[0]  # only liked movies the group could be offered
        row = {"group_id": gid, "size": len(members), "popularity list": np.median(pop[top]),
               "movies members liked in test": np.median(pop[liked_items]), "all candidates": np.median(pop[cand])}
        for model in eg.MODELS:
            t, _ = eg.top_k(eg.AGGREGATORS["average"](scores[model][r_all]), cand, k, by_id)
            row[model + " list"] = np.median(pop[t])
        contents.append(row)
    return pd.DataFrame(rows), pd.DataFrame(contents)


def vs_popularity(bg, base, rng):
    """Model minus popularity per group (mean, 95% bootstrap CI) and the share of groups each side wins."""
    sub = bg[bg["tie"] == "movieId"]
    ref = base.set_index("group_id")
    out = []
    for agg in AGGS:
        for model in eg.MODELS:
            m = sub[(sub["agg"] == agg) & (sub["model"] == model)].set_index("group_id")
            for metric in ("p_mean", "ndcg_mean"):
                d = (m[metric] - ref["popularity_" + metric]).round(9)
                for z in (2, 3, 4, 5, "all"):
                    x = (d if z == "all" else d[m["size"] == z]).to_numpy()
                    mean, lo, hi = boot_mean_ci(x, rng)
                    out.append({"agg": agg, "model": model, "metric": metric, "size": str(z), "n_groups": len(x),
                                "model_minus_popularity": mean, "ci_low": lo, "ci_high": hi,
                                "model_better_pct": (x > 0).mean() * 100, "tie_pct": (x == 0).mean() * 100,
                                "popularity_better_pct": (x < 0).mean() * 100})
    return pd.DataFrame(out)


def fig_reference_ladder(bg, base, k):
    sub = bg[(bg["tie"] == "movieId") & (bg["agg"] == "average")]
    mean = {m: sub[sub["model"] == m]["p_mean"].mean() for m in eg.MODELS}
    rand = base["random_p_mean"].mean()
    diag = os.path.join(OUT, "group_eval_diag_mf.csv")  # written by eval_group_diag_mf.py (optional)
    extra = {}
    if os.path.exists(diag):
        d = pd.read_csv(diag)
        d = d[(d["size"].astype(str) == "all") & d["agg"].isin(["average", "-"])]
        extra = {s: d[d["system"] == s]["p_mean"].iloc[0] for s in ("shrunk_item_mean", "mf_bias_only")}
    bars = [("Upper bound (best possible shared list)", base["upper_bound_group_p_mean"].mean(), None, "bound"),
            ("Popularity (most-rated movies)", base["popularity_p_mean"].mean(), None, "popularity")]
    if extra:
        bars.append(("Item mean rating, shrunk (no personalisation)", extra["shrunk_item_mean"], None, "item_mean"))
    bars.append(("MF", mean["mf"], None, "mf"))
    if extra:
        bars.append(("MF biases only (no latent part)", extra["mf_bias_only"], None, "mf_bias"))
    bars += [("Content-based", mean["content"], None, "content"), ("Random", rand, None, "random")]
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(bars) + 1.6))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    fills = {"bound": None, "popularity": REF, "item_mean": REF, "mf": COLOR["mf"], "mf_bias": "#86b6ef",  # blue ramp, step 250
             "content": COLOR["content"], "random": AXIS}
    for y, (label, v, _, key) in enumerate(bars):
        if fills[key] is None:
            ax.barh(y, v, height=0.46, facecolor="none", edgecolor=INK2, linewidth=1.4, zorder=3)
        else:
            ax.barh(y, v, height=0.46, color=fills[key], zorder=3)
        note = "" if key in ("bound", "random") else "   %.1f× random" % (v / rand)
        ax.text(v + 0.006, y, "%.4f%s" % (v, note), va="center", fontsize=10.5, color=INK2)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.45)
    style_axes(ax, None, "Precision@%d (mean over members), all 4,000 groups, Average aggregation for the models" % k)
    save(fig, "fig8_reference_ladder.png", "Both models beat random, but sit below simple non-personalised lists",
         "A group's one shared Top-%d list, graded against each member's own liked test movies." % k)
    return pd.DataFrame([{"list": b[0], "p_mean": b[1]} for b in bars])


def fig_reference_by_size(bg, base, k):
    sub = bg[(bg["tie"] == "movieId") & (bg["agg"] == "average")]
    labels = ["2", "3", "4", "5", "All"]

    def by_size(frame, col):
        s = frame.groupby("size")[col].mean()
        return [s[int(z)] if z != "All" else frame[col].mean() for z in labels]

    series = [("Random", by_size(base, "random_p_mean"), AXIS),
              ("Content-based", by_size(sub[sub["model"] == "content"], "p_mean"), COLOR["content"]),
              ("MF", by_size(sub[sub["model"] == "mf"], "p_mean"), COLOR["mf"]),
              ("Popularity", by_size(base, "popularity_p_mean"), REF)]
    bound = by_size(base, "upper_bound_group_p_mean")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x, w, gap = np.arange(5), 0.17, 0.015
    for j, (name, vals, colr) in enumerate(series):
        ax.bar(x + (j - 1.5) * (w + 2 * gap), vals, w, color=colr, zorder=3)
    for j, (_, vals, _) in enumerate(series):  # label only the pooled cluster
        ax.text(x[-1] + (j - 1.5) * (w + 2 * gap), vals[-1] + 0.003, "%.3f" % vals[-1], ha="center", fontsize=9.5,
                color=INK2)
    ax.set_xticks(x, ["Size %s" % z if z != "All" else "All sizes" for z in labels])
    style_axes(ax, "Precision@%d (mean over members)" % k)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in series]
    save(fig, "fig9_reference_by_size.png", "The ranking holds at every group size: popularity > MF > content-based > random",
         "Same four lists at each group size. The upper bound for a shared list falls from %.2f (size 2) to %.2f "
         "(size 5), so larger groups cap every list." % (bound[0], bound[3]),
         legend=(handles, [s[0] for s in series]))


def fig_list_popularity(contents):
    cols = [("Popularity list", "popularity list", REF), ("Liked in test (inside the pool)", "movies members liked in test", None),
            ("MF list", "mf list", COLOR["mf"]), ("Content-based list", "content list", COLOR["content"]),
            ("All candidates", "all candidates", AXIS)]
    vals = [contents[c].mean() for _, c, _ in cols]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    for y, ((label, _, colr), v) in enumerate(zip(cols, vals)):
        if colr is None:
            ax.barh(y, v, height=0.46, facecolor="none", edgecolor=INK2, linewidth=1.4, zorder=3)
        else:
            ax.barh(y, v, height=0.46, color=colr, zorder=3)
        ax.text(v + 25, y, "%s" % format(int(round(v)), ","), va="center", fontsize=10.5, color=INK2)
    ax.set_yticks(range(len(cols)), [c[0] for c in cols])
    ax.invert_yaxis()
    ax.set_xlim(0, 2000)
    style_axes(ax, None, "Median train-rating count of the movies in each list, averaged over 4,000 groups")
    save(fig, "fig10_what_the_lists_contain.png",
         "MF and content-based lists hold less popular movies than the ones people liked",
         "Movies that members liked in the test split tend to be widely rated; the models' lists are not.")
    return pd.DataFrame({"list": [c[0] for c in cols], "median_train_ratings": vals})


def pick_cases(bg):
    a = bg[(bg["tie"] == "movieId") & (bg["agg"] == "average")].pivot(index="group_id", columns="model", values="p_mean")
    a = a.round(9)  # ties between groups resolve to the smallest group_id
    typical = ((a["mf"] - a["mf"].mean()) ** 2 + (a["content"] - a["content"].mean()) ** 2).round(12).idxmin()
    return [("Near-average group", "closest to the average (MF, content-based) P@10 pair", int(typical)),
            ("Largest MF win", "largest P@10 gap in MF's favour", int((a["mf"] - a["content"]).round(9).idxmax())),
            ("Largest content-based win", "largest P@10 gap in content-based's favour",
             int((a["content"] - a["mf"]).round(9).idxmax()))]


def short(title, n=42):
    """Shorten the name but keep the release year (MovieLens titles end with ' (YYYY)')."""
    if len(title) <= n:
        return title
    return title[:-7][:n - 8] + "…" + title[-7:]


def fig_case(num, label, rule, gid, members, ctx, k):
    scores, user_ids, items, seen, rel, titles, vectors, genres, n_train = ctx
    r = eg.locate(user_ids, members)
    n_liked = rel.sum(axis=1)
    cand = ~seen[r].any(axis=0)
    letters = "ABCDE"
    height = 5.6
    fig = plt.figure(figsize=(14, height))
    ax = fig.add_axes((0, 0, 1, 1 - 1.0 / height))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.text(1, 94, "Group members", fontsize=11.5, fontweight="semibold", color=INK)
    y = 88
    for i, u in enumerate(r):
        top_g = [genres[j] for j in np.argsort(-vectors[u])[:2]]
        note = "" if n_liked[u] > 0 else "  (no liked test movie: not graded)"
        ax.text(1, y, "%s  user %d" % (letters[i], user_ids[u]), fontsize=10.5, color=INK, fontweight="semibold")
        ax.text(1, y - 3.6, "%d train ratings · %d liked in test%s" % (n_train[u], n_liked[u], note),
                fontsize=9.5, color=INK2)
        ax.text(1, y - 7.0, "favourite genres: %s" % ", ".join(top_g), fontsize=9.5, color=INK2)
        y -= 13.5

    summary = []
    for col, model, x0 in ((0, "mf", 33), (1, "content", 67)):
        group_score = eg.AGGREGATORS["average"](scores[model][r])
        top, _ = eg.top_k(group_score, cand, k, np.arange(len(items)))
        valid = n_liked[r] > 0
        hit = rel[r][:, top]
        p, n = eg.member_metrics(rel[r[valid]], n_liked[r[valid]], top, k)
        summary.append((model, p, n))
        ax.text(x0, 94, "%s Top-%d (Average)" % (MODEL_LABEL[model], k), fontsize=11.5, fontweight="semibold",
                color=INK)
        ax.plot([x0 - 0.6, x0 - 0.6], [91.5, 94.8], color=COLOR[model], linewidth=3, solid_capstyle="round")
        for rank, item in enumerate(top):
            yy = 88 - rank * 6.4
            who = ", ".join(letters[i] for i in range(len(r)) if hit[i, rank])
            if who:
                ax.add_patch(plt.Rectangle((x0 - 1, yy - 2.6), 31.5, 5.6, color=COLOR[model], alpha=0.14, linewidth=0))
            ax.text(x0, yy, "%2d" % (rank + 1), fontsize=9.5, color=MUTED, va="center")
            ax.text(x0 + 2.6, yy, short(titles[items[item]]), fontsize=9.8, color=INK, va="center")
            ax.text(x0 + 29.5, yy, who, fontsize=9.8, color=INK, va="center", ha="right", fontweight="semibold")
    line = "   |   ".join("%s: members' P@%d %s, mean %.3f" % (
        MODEL_LABEL[m], k, " ".join("%.1f" % v for v in p), p.mean()) for m, p, _ in summary)
    ax.text(1, 12, line + "     (letters on a highlighted row = members who liked that movie in test)", fontsize=9.5,
            color=INK2)
    header(fig, "Case %d · %s (group %d, %d members)" % (num, label, gid, len(members)),
           "Selection rule: %s (ties: smallest group_id)." % rule)
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, "fig7_case%d_%s.png" % (num, label.lower().replace(" ", "_").replace("-", "_")))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved", path)
    return {"case": num, "label": label, "rule": rule, "group_id": gid, "members": [int(user_ids[u]) for u in r],
            "graded": [bool(n_liked[u] > 0) for u in r],  # *_p_members lists only the graded members, in this order
            **{"%s_p_members" % m: [round(float(v), 3) for v in p] for m, p, _ in summary}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mf-scores", default=os.path.join(RESULTS, "mf_scores.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    setup_style()
    rng = np.random.RandomState(args.seed)
    with open(os.path.join(OUT, "group_eval_meta.json")) as f:
        meta = json.load(f)
    k = meta["k"]

    print("[1/5] load split, groups and scores")
    train = load_ratings(os.path.join(RESULTS, "ratings_train.csv"))
    test = load_ratings(os.path.join(RESULTS, "ratings_test_eval.csv"))
    movies = pd.read_csv(os.path.join(RESULTS, "movies_processed.csv"), encoding="utf-8-sig")
    user_ids, items, seen, rel = eg.build_index(train, test)
    gdf = pd.read_csv(os.path.join(OUT, "group_eval_groups.csv"))
    groups = [np.array(s.split("|"), dtype=np.int64) for s in gdf["userIds"]]
    expected = eg.sample_groups(user_ids, meta["groups_per_size"], meta["seed"])
    assert all(np.array_equal(a, b) for a, b in zip(groups, expected)), "groups differ from the evaluated ones"
    bg = pd.read_csv(os.path.join(OUT, "group_eval_by_group.csv"))
    diff = pd.read_csv(os.path.join(OUT, "group_eval_diff.csv"))
    mf_mat, rmse = eg.load_mf(args.mf_scores, user_ids, items, seen, test)
    assert abs(rmse - meta["mf_test_rmse_from_scores_file"]) < 1e-9
    scores = {"mf": mf_mat, "content": eg.load_content(np.unique(np.concatenate(groups)), user_ids, items)}

    print("[2/5] individual Top-%d reference" % k)
    ind = individual_reference(groups, scores, user_ids, seen, rel, k)
    ind.to_csv(os.path.join(OUT, "group_eval_individual.csv"), index=False)
    ind2, loss = individual_same_pool(groups, scores, user_ids, seen, rel, k)
    ind2.to_csv(os.path.join(OUT, "group_eval_individual_samepool.csv"), index=False)
    loss.to_csv(os.path.join(OUT, "group_eval_candidate_loss.csv"), index=False)
    print(loss.round(3).to_string(index=False))

    print("[3/5] cohesion")
    vectors, genres = genre_vectors(movies, items, train, user_ids)
    coh = cohesion(groups, vectors, user_ids)
    sizes = np.array([len(g) for g in groups])
    tercile = tercile_labels(coh, sizes)
    coh_tbl = pd.DataFrame({"group_id": range(len(groups)), "size": sizes, "cohesion": coh, "tercile": tercile})
    coh_tbl.to_csv(os.path.join(OUT, "group_eval_cohesion.csv"), index=False)
    cutoffs = coh_tbl.dropna().groupby(["size", "tercile"])["cohesion"].agg(["min", "max", "count"])
    print(cutoffs.round(3).to_string())
    coh_df = bg[bg["tie"] == "movieId"].merge(coh_tbl[["group_id", "tercile"]], on="group_id").dropna(subset=["tercile"])

    print("[4/5] figures")
    summary = fig_context(ind, bg, k, ind2)
    summary.to_csv(os.path.join(OUT, "group_eval_individual_summary.csv"), index=False)
    fig_distribution(bg, k)
    paired = fig_paired(bg, k)
    paired.to_csv(os.path.join(OUT, "group_eval_paired_summary.csv"), index=False)
    fig_forest(diff)
    fig_cohesion(coh_df, k, rng).to_csv(os.path.join(OUT, "group_eval_cohesion_summary.csv"), index=False)
    fig_lm_vs_avg(coh_df, rng, bg).to_csv(os.path.join(OUT, "group_eval_cohesion_lm_vs_avg.csv"), index=False)
    base, contents = reference_levels(groups, scores, user_ids, items, seen, rel, train, k)
    base.to_csv(os.path.join(OUT, "group_eval_baselines.csv"), index=False)
    fig_cohesion_lift(coh_df, base, k).to_csv(os.path.join(OUT, "group_eval_cohesion_lift.csv"), index=False)
    vs_popularity(bg, base, rng).to_csv(os.path.join(OUT, "group_eval_vs_popularity.csv"), index=False)
    fig_reference_ladder(bg, base, k).to_csv(os.path.join(OUT, "group_eval_reference_ladder.csv"), index=False)
    fig_reference_by_size(bg, base, k)
    fig_list_popularity(contents).to_csv(os.path.join(OUT, "group_eval_list_popularity.csv"), index=False)

    print("[5/5] case studies")
    titles = dict(zip(movies["movieId"], movies["title"]))
    n_train = np.bincount(eg.locate(user_ids, train["userId"].to_numpy()), minlength=len(user_ids))
    ctx = (scores, user_ids, items, seen, rel, titles, vectors, genres, n_train)
    cases = [fig_case(i + 1, label, rule, gid, groups[gid], ctx, k) for i, (label, rule, gid) in enumerate(pick_cases(bg))]
    with open(os.path.join(FIGURES, "cases.json"), "w") as f:
        json.dump(cases, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
