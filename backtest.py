"""Replay the weekly update loop over a past season window.

Runs exactly the same per-gameweek procedure as update_model.py (train on matches
before the gameweek, predict it from pre-gameweek features, score it) for every
gameweek that starts inside the window, and writes the results in tracking.csv
format. Used to show what the live Oct-Nov 2026 record looks like, using Oct-Nov 2025.

Usage:
    python backtest.py                                   # Oct-Nov 2025 (default)
    python backtest.py --season 2024-25 --start 2024-10-01 --end 2024-11-30
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.features import build_dataset
from src.gameweeks import assign_gameweeks
from src.ingest import ROOT
from update_model import TRACKING_COLS, load_best_params, score_gameweek, tracking_rows


def run(season: str, start: str, end: str) -> pd.DataFrame:
    data = build_dataset()
    params = load_best_params()
    gws = assign_gameweeks(data[(data["Season"] == season) & data["played"]])
    span = gws.groupby("gameweek")["Kickoff"].agg(["min", "max"])
    todo = span[(span["min"] >= start) & (span["max"] < pd.Timestamp(end) + pd.Timedelta(days=1))].index
    print(f"Backtesting {season}, {start} to {end}: {len(todo)} gameweeks")

    rows = []
    for gw in todo:
        gw_rows = gws[gws["gameweek"] == gw]
        metrics, _, n_train = score_gameweek(data, gw_rows, params)
        rows += tracking_rows(season, int(gw), gw_rows, metrics, n_train)
    return pd.DataFrame(rows, columns=TRACKING_COLS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", default="2025-26")
    p.add_argument("--start", default="2025-10-01")
    p.add_argument("--end", default="2025-11-30")
    args = p.parse_args()

    result = run(args.season, args.start, args.end)
    out = ROOT / "data" / f"backtest_{args.season.replace('-', '_')}.csv"
    result.to_csv(out, index=False)
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
