import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
REPS = (5, 10, 25, 50)
PADS = (25, 50, 100, 200)
METHODS = ("cosine", "pivoted")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COLOR = {"cosine": "#2a78d6", "pivoted": "#eb6834"}
MARKER = {"cosine": "o", "pivoted": "s"}
SEQ = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "cos_piv", ["#184f95", "#86b6ef", "#f0efec", "#f3a586", "#eb6834"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def value(rows, key, **conds):
    for r in rows:
        if all(r[k] == str(v) for k, v in conds.items()):
            return float(r[key])
    raise KeyError(conds)


def lines(ax, xs, series, xlabels, ylabel, title, ylim=None, label_ends=True):
    # Pivoted is drawn underneath and thicker so both lines stay visible where they overlap
    style = {"pivoted": dict(lw=2.4, ms=7, zorder=3), "cosine": dict(lw=1.4, ms=4.8, zorder=4)}
    for m in ("pivoted", "cosine"):
        ax.plot(xs, series[m], color=COLOR[m], marker=MARKER[m], markeredgecolor=SURFACE,
                markeredgewidth=1.0, label=m, **style[m])
    ax.set_xticks(xs, xlabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlim(xs[0] - 0.3, xs[-1] + 0.9)
    if ylim:
        ax.set_ylim(*ylim)
    if label_ends:
        lo, hi = ax.get_ylim()
        gap = 0.07 * (hi - lo)
        ends = sorted(((series[m][-1], m) for m in METHODS))
        # Keep the two end labels from overlapping
        ys = [ends[0][0], max(ends[1][0], ends[0][0] + gap)]
        for (_, m), y in zip(ends, ys):
            ax.annotate(m, (xs[-1], y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_2)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=ROOT / "results" / "attacks")
    parser.add_argument("--out", default=ROOT / "docs" / "figures")
    parser.add_argument("--name", default="attack", help="file name prefix")
    args = parser.parse_args(argv)
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    levels = read(res / "levels.csv")
    groups = read(res / "by_group.csv")
    summary = json.loads((res / "summary.json").read_text(encoding="utf-8"))
    n_queries = len(summary["query_terms"])
    # MAP covers only the queries with a relevant article
    n_eval = n_queries - len(summary.get("queries_without_relevant", []))
    n_targets = int(value(levels, "trials", attack="repeat x5", method="cosine"))
    n_long = int(value(groups, "trials", attack="repeat x5", method="cosine", length_group="long"))
    xs = list(range(4))
    rep_labels = [f"×{k}" for k in REPS]
    pad_labels = [f"{p}%" for p in PADS]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
    for ax, grp, title in ((axes[0], None, f"All targets (n = {n_targets})"),
                           (axes[1], "long", f"Long articles only (n = {n_long})")):
        series = {m: [value(levels, "success_pct", attack=f"repeat x{k}", method=m) if grp is None else
                      value(groups, "success_pct", attack=f"repeat x{k}", method=m, length_group=grp)
                      for k in REPS] for m in METHODS}
        lines(ax, xs, series, rep_labels, "entered top 10 (%)" if grp is None else "", title, ylim=(75, 102))
        ax.set_xlabel("query terms appended")
    fig.suptitle("Keyword repetition: share of edited articles that enter the top 10",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, out, f"{args.name}_repetition")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    ratio = {m: [value(levels, "mean_score_ratio", attack=f"pad {p}%", method=m) for p in PADS] for m in METHODS}
    drank = {m: [value(levels, "median_rank_change", attack=f"pad {p}%", method=m) for p in PADS] for m in METHODS}
    lines(axes[0], xs, ratio, pad_labels, "new score / old score",
          "Score kept (targets retrieved before the edit)", ylim=(0.5, 1.0))
    lines(axes[1], xs, drank, pad_labels, "median rank change", "Rank change (all targets; negative = down)")
    axes[1].axhline(0, color=AXIS, lw=0.8, zorder=1)
    for ax in axes:
        ax.set_xlabel("padding, % of original length")
    fig.suptitle("Unrelated padding: no trial entered the top 10 under either method",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, out, f"{args.name}_padding")

    grid = {m: [[value(levels, "success_pct", attack=f"repeat x{k} + pad {p}%", method=m) for p in PADS]
                for k in REPS] for m in METHODS}
    diff = [[grid["pivoted"][i][j] - grid["cosine"][i][j] for j in range(4)] for i in range(4)]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    for ax, data, title, cmap, norm, fmt in (
            (axes[0], grid["cosine"], "Cosine: entered top 10 (%)", SEQ, plt.Normalize(50, 100), "{:.0f}"),
            (axes[1], grid["pivoted"], "Pivoted: entered top 10 (%)", SEQ, plt.Normalize(50, 100), "{:.0f}"),
            (axes[2], diff, f"Pivoted − cosine (points; 1 target = {100 / n_targets:.1f})",
             DIV, TwoSlopeNorm(0, -10, 10), "{:+.1f}")):
        ax.imshow(data, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(4), pad_labels)
        ax.set_yticks(range(4), rep_labels)
        ax.set_xlabel("padding")
        ax.set_title(title, loc="left", color=INK)
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks([x + 0.5 for x in range(3)], minor=True)
        ax.set_yticks([y + 0.5 for y in range(3)], minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        for i in range(4):
            for j in range(4):
                v = data[i][j]
                dark = norm(v) > 0.62 if cmap is SEQ else abs(v) > 7
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                        color=SURFACE if dark else INK)
    axes[0].set_ylabel("repetitions")
    fig.suptitle(f"Combined attack: repetition plus padding ({n_targets} targets per cell)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, out, f"{args.name}_combined")

    diag = [f"repeat x{k} + pad {p}%" for k, p in zip(REPS, PADS)]
    panels = (
        ("Repetition", [f"repeat x{k}" for k in REPS], rep_labels),
        ("Padding", [f"pad {p}%" for p in PADS], pad_labels),
        ("Combined (diagonal)", diag, [f"×{k}+{p}%" for k, p in zip(REPS, PADS)]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), sharey=True)
    for i, (ax, (title, labels, ticks)) in enumerate(zip(axes, panels)):
        series = {m: [value(levels, "mean_d_map", attack=a, method=m) for a in labels] for m in METHODS}
        lines(ax, xs, series, ticks, f"mean ΔMAP ({n_eval} queries)" if i == 0 else "", title,
              ylim=(-0.026, 0.004), label_ends=(i == 2))
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        if i == 2:
            ax.tick_params(axis="x", labelsize=7.5)
    fig.suptitle("Change in MAP from one edited article (vs each method's clean MAP)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, out, f"{args.name}_quality")

    names = ["repetition", "padding", "combined", "quality"]
    # Only runs with queries that have no relevant article write this file
    if (res / "by_query_type.csv").exists():
        types = read(res / "by_query_type.csv")
        n_with = int(value(types, "trials", attack="repeat x5", method="cosine", query_has_relevant=1))
        n_without = int(value(types, "trials", attack="repeat x5", method="cosine", query_has_relevant=0))
        # Solid lines for queries with a relevant article, dashed for queries without
        kind = {1: dict(ls="-", mfc=None), 0: dict(ls=(0, (3, 2)), mfc=SURFACE)}
        tag = {1: f"has relevant (n = {n_with})", 0: f"no relevant (n = {n_without})"}
        panels = (
            ("Repetition: entered top 10 (%)", [f"repeat x{k}" for k in REPS], rep_labels,
             "success_pct", (90, 101.5), "query terms appended"),
            ("Combined (diagonal): entered top 10 (%)", diag, [f"×{k}+{p}%" for k, p in zip(REPS, PADS)],
             "success_pct", (90, 101.5), "repetitions + padding"),
            ("Padding: median rank change", [f"pad {p}%" for p in PADS], pad_labels,
             "median_rank_change", None, "padding, % of original length"),
        )
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
        for ax, (title, attacks, ticks, key, ylim, xlabel) in zip(axes, panels):
            for has in (1, 0):
                for m in ("pivoted", "cosine"):
                    ys = [value(types, key, attack=a, method=m, query_has_relevant=has) for a in attacks]
                    ax.plot(xs, ys, color=COLOR[m], marker=MARKER[m], ls=kind[has]["ls"],
                            lw=2.0 if m == "pivoted" else 1.3, ms=6.5 if m == "pivoted" else 4.5,
                            markeredgecolor=COLOR[m] if has == 0 else SURFACE, markeredgewidth=1.0,
                            markerfacecolor=kind[has]["mfc"] or COLOR[m],
                            zorder=3 if m == "pivoted" else 4, label=f"{m}, {tag[has]}")
            ax.set_xticks(xs, ticks)
            ax.set_xlim(xs[0] - 0.3, xs[-1] + 0.3)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_title(title, loc="left", color=INK)
            ax.set_xlabel(xlabel)
            if key == "median_rank_change":
                ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        axes[1].tick_params(axis="x", labelsize=7.5)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.1))
        fig.suptitle("Queries with a relevant article (solid) vs queries with none (dashed, open markers)",
                     x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout()
        save(fig, out, f"{args.name}_query_type")
        names.append("query_type")

    print(f"wrote {out}/{args.name}_{{{','.join(names)}}}.{{png,pdf}}")


if __name__ == "__main__":
    main()
