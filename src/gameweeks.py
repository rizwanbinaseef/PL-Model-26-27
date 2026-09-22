"""Group a season's fixtures into gameweeks.

football-data.co.uk has no gameweek column, so matches are grouped into the EPL's
scheduling slots: a weekend slot (Fri-Mon) and a midweek slot (Tue-Thu). Slots smaller
than `min_size` (e.g. a lone rearranged fixture) are merged into the next slot, so a
model is never evaluated on just 1-2 matches.
"""
from __future__ import annotations

import pandas as pd


def slot_start(day: pd.Series) -> pd.Series:
    """First day of the scheduling slot: the Friday for Fri-Mon, the Tuesday for Tue-Thu."""
    wd = day.dt.weekday  # Mon=0 .. Sun=6
    offset = wd.map({0: 3, 1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2})
    return day - pd.to_timedelta(offset, unit="D")


def assign_gameweeks(season: pd.DataFrame, min_size: int = 5) -> pd.DataFrame:
    """Add `gameweek` (1..n) and `gw_pending_merge` columns. Input must be one season,
    including unplayed fixtures, with a Kickoff column."""
    df = season.sort_values(["Kickoff", "HomeTeam"], kind="stable").copy()
    block = slot_start(df["Kickoff"].dt.normalize())

    gw, pending, current = {}, [], 0
    sizes = block.value_counts().sort_index()
    for b, size in sizes.items():
        pending.append(b)
        if sum(sizes[p] for p in pending) >= min_size:
            current += 1
            gw.update({p: current for p in pending})
            pending = []
    # a trailing small block waits for the next one before it can be treated as complete
    df["gw_pending_merge"] = block.isin(pending)
    gw.update({p: current + 1 for p in pending})
    df["gameweek"] = block.map(gw).astype(int)
    return df


def completed_gameweeks(season: pd.DataFrame, as_of: pd.Timestamp) -> list[int]:
    """Gameweeks whose matches are all played and whose last match day is before `as_of`."""
    g = season.groupby("gameweek").agg(
        all_played=("played", "all"),
        last_day=("Kickoff", lambda s: s.max().normalize()),
        pending=("gw_pending_merge", "any"),
    )
    done = g["all_played"] & ~g["pending"] & (g["last_day"] < as_of.normalize())
    return sorted(g.index[done].tolist())
