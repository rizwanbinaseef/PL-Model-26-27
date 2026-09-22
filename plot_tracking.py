"""Plot model performance per gameweek from data/tracking.csv.

Top panel: accuracy; bottom panel: log loss. Faint dots are single gameweeks
(10 matches each, so very noisy); the lines are cumulative, match-weighted
averages over the plotted window, which is where any real trend shows up.

Usage:
    python plot_tracking.py                       # Oct-Nov 2026 (default window)
    python plot_tracking.py --all                 # whole season so far
    python plot_tracking.py --start 2026-10-01 --end 2026-11-30 --out reports/x.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
TRACKING = ROOT / "data" / "tracking.csv"

SERIES = {  # fixed order -> fixed colour (validated categorical palette)
    "logreg": ("Logistic regression", "#2a78d6"),
    "xgboost": ("XGBoost", "#eb6834"),
    "bookmaker": ("Bookmaker closing odds", "#1baf7a"),
}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def cumulative(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("gameweek").copy()
    w = df["n_matches"]
    df["cum_accuracy"] = (df["accuracy"] * w).cumsum() / w.cumsum()
    df["cum_log_loss"] = (df["log_loss"] * w).cumsum() / w.cumsum()
    return df


def _label_line_ends(ax, points, min_gap_frac=0.06):
    """Value labels at line ends, nudged vertically so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    placed = []
    for x, y in sorted(points, key=lambda p: p[1]):
        ty = max(y, placed[-1] + gap) if placed else y
        placed.append(ty)
        ax.annotate(f"{y:.3f}", (x, y), xytext=(x + 0.12, ty), textcoords="data",
                    va="center", fontsize=9, color=INK)


def plot(tracking: pd.DataFrame, out: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor=SURFACE)
    panels = [("accuracy", "cum_accuracy", "Accuracy (higher is better)"),
              ("log_loss", "cum_log_loss", "Log loss (lower is better)")]
    gws = sorted(tracking["gameweek"].unique())
    for ax, (raw, cum, label) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        end_labels = []
        for key, (name, color) in SERIES.items():
            d = tracking[tracking["model"] == key]
            if d.empty:
                continue
            d = cumulative(d)
            ax.scatter(d["gameweek"], d[raw], s=28, color=color, alpha=0.3, linewidths=0, zorder=2)
            ax.plot(d["gameweek"], d[cum], color=color, lw=2, marker="o", ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
            end_labels.append((d["gameweek"].iloc[-1], d[cum].iloc[-1]))
        _label_line_ends(ax, end_labels)
        ax.set_ylabel(label, color=INK, fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)

    first = tracking.drop_duplicates("gameweek").set_index("gameweek")["start_date"]
    axes[1].set_xticks(gws)
    axes[1].set_xticklabels([f"GW{g}\n{pd.Timestamp(first[g]):%d %b}" for g in gws])
    axes[1].set_xlim(min(gws) - 0.4, max(gws) + 0.8)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, y=0.985)
    fig.text(0.02, 0.945, "Lines: cumulative (match-weighted). Faint dots: single gameweek.",
             fontsize=9, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.015, 0.93), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2026-10-01")
    p.add_argument("--end", default="2026-11-30")
    p.add_argument("--all", action="store_true", help="ignore --start/--end, plot every logged gameweek")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "tracking_oct_nov_2026.png")
    args = p.parse_args()

    df = pd.read_csv(TRACKING, parse_dates=["start_date", "end_date"])
    if args.all:
        window, title = df, "EPL 2026-27: out-of-sample performance by gameweek"
        if args.out.name == "tracking_oct_nov_2026.png":
            args.out = args.out.with_name("tracking_season_2026_27.png")
    else:
        window = df[(df["start_date"] >= args.start) & (df["end_date"] <= args.end)]
        title = f"EPL 2026-27: out-of-sample performance, {pd.Timestamp(args.start):%d %b} to {pd.Timestamp(args.end):%d %b %Y}"
    if window.empty:
        sys.exit(f"No gameweeks logged between {args.start} and {args.end} yet "
                 f"({len(df['gameweek'].unique())} logged in total). Use --all to plot everything.")

    plot(window, args.out, title)
    summary = window.groupby("model").apply(
        lambda d: pd.Series({"gameweeks": d["gameweek"].nunique(), "matches": d["n_matches"].sum(),
                             "accuracy": (d["accuracy"] * d["n_matches"]).sum() / d["n_matches"].sum(),
                             "log_loss": (d["log_loss"] * d["n_matches"]).sum() / d["n_matches"].sum()}),
        include_groups=False)
    print(summary.round(3).to_string())
    print(f"\nSaved {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
