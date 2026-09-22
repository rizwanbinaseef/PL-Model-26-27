"""Leakage-free feature engineering.

Matches are processed in kickoff order. For each match we FIRST read the current
team state (Elo, recent results) to build its feature row, and only THEN update
the state with that match's result. A match's features therefore depend only on
matches that kicked off before it. Rows without a result (future fixtures) get
features but never update the state.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

RESULT_TO_LABEL = {"H": 0, "D": 1, "A": 2}
LABELS = ["H", "D", "A"]


@dataclass(frozen=True)
class EloConfig:
    base: float = 1500.0
    k: float = 20.0
    home_advantage: float = 60.0
    season_regression: float = 0.2   # fraction pulled back toward `base` between seasons
    promoted_rating: float = 1400.0  # start rating for teams first seen after season one


FORM_WINDOW = 5
REST_CAP_DAYS = 21

FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff", "elo_home_expected",
    "home_form_pts", "home_form_gf", "home_form_ga",
    "away_form_pts", "away_form_gf", "away_form_ga",
    "home_home_pts", "home_home_gf", "home_home_ga",
    "away_away_pts", "away_away_gf", "away_away_ga",
    "home_rest_days", "away_rest_days", "rest_diff",
]


def elo_expected(r_home: float, r_away: float, home_adv: float) -> float:
    """Expected score for the home team (win=1, draw=0.5)."""
    return 1.0 / (1.0 + 10 ** ((r_away - (r_home + home_adv)) / 400.0))


def _margin_multiplier(goal_diff: int) -> float:
    return math.log(abs(goal_diff) + 1) + 1.0 if goal_diff else 1.0


def _points(gf: int, ga: int) -> int:
    return 3 if gf > ga else (1 if gf == ga else 0)


def _window_stats(window: deque) -> tuple[float, float, float]:
    """Mean (points, goals for, goals against) over the window; NaN if empty."""
    if not window:
        return (np.nan, np.nan, np.nan)
    arr = np.asarray(window, dtype=float)
    return tuple(arr.mean(axis=0))


def build_features(matches: pd.DataFrame, elo: EloConfig = EloConfig()) -> pd.DataFrame:
    """Return `matches` (sorted by kickoff) with FEATURE_COLUMNS and `target` added.

    Required columns: Season, Kickoff, HomeTeam, AwayTeam, FTHG, FTAG (NaN for unplayed).
    """
    df = matches.sort_values(["Kickoff", "HomeTeam"], kind="stable").reset_index(drop=True)

    ratings: dict[str, float] = {}
    form = defaultdict(lambda: deque(maxlen=FORM_WINDOW))       # all venues
    home_form = defaultdict(lambda: deque(maxlen=FORM_WINDOW))  # home games only
    away_form = defaultdict(lambda: deque(maxlen=FORM_WINDOW))  # away games only
    last_played: dict[str, pd.Timestamp] = {}
    first_season = df["Season"].iloc[0] if len(df) else None
    current_season = first_season

    rows = []
    for m in df.itertuples(index=False):
        # New season: regress every known rating toward the mean.
        if m.Season != current_season:
            for t in ratings:
                ratings[t] += elo.season_regression * (elo.base - ratings[t])
            current_season = m.Season

        h, a = m.HomeTeam, m.AwayTeam
        for t in (h, a):
            if t not in ratings:
                ratings[t] = elo.base if m.Season == first_season else elo.promoted_rating

        # ---- 1. features from state BEFORE this match ----
        rh, ra = ratings[h], ratings[a]
        p_home = elo_expected(rh, ra, elo.home_advantage)
        hf, af = _window_stats(form[h]), _window_stats(form[a])
        hh, aa = _window_stats(home_form[h]), _window_stats(away_form[a])
        rest_h = _rest_days(last_played.get(h), m.Kickoff)
        rest_a = _rest_days(last_played.get(a), m.Kickoff)
        rows.append([rh, ra, rh - ra, p_home, *hf, *af, *hh, *aa,
                     rest_h, rest_a, rest_h - rest_a])

        # ---- 2. update state with this match's result (played matches only) ----
        if pd.isna(m.FTHG) or pd.isna(m.FTAG):
            continue
        hg, ag = int(m.FTHG), int(m.FTAG)
        actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = elo.k * _margin_multiplier(hg - ag) * (actual - p_home)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

        form[h].append((_points(hg, ag), hg, ag))
        form[a].append((_points(ag, hg), ag, hg))
        home_form[h].append((_points(hg, ag), hg, ag))
        away_form[a].append((_points(ag, hg), ag, hg))
        last_played[h] = last_played[a] = m.Kickoff

    feats = pd.DataFrame(rows, columns=FEATURE_COLUMNS, index=df.index)
    out = pd.concat([df, feats], axis=1)
    if "FTR" in out.columns:
        out["target"] = out["FTR"].map(RESULT_TO_LABEL)
    return out


def _rest_days(last: pd.Timestamp | None, now: pd.Timestamp) -> float:
    if last is None:
        return float(REST_CAP_DAYS)
    return float(min((now - last).total_seconds() / 86400.0, REST_CAP_DAYS))


def build_dataset(include_fixtures: bool = True) -> pd.DataFrame:
    """Load raw data, append unplayed fixtures, compute features, save to data/processed."""
    from src.ingest import CURRENT_SEASON, RAW_DIR, ROOT, load_matches, season_label

    matches = load_matches()
    fx_path = RAW_DIR / f"fixtures_{CURRENT_SEASON}.csv"
    if include_fixtures and fx_path.exists():
        fx = pd.read_csv(fx_path)
        if len(fx):
            fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, format="mixed")
            fx["Kickoff"] = pd.to_datetime(fx["Date"].dt.strftime("%Y-%m-%d") + " " + fx["Time"].fillna("15:00"))
            fx["Season"] = season_label(CURRENT_SEASON)
            fx["match_id"] = fx["Date"].dt.strftime("%Y%m%d") + "_" + fx["HomeTeam"] + "_" + fx["AwayTeam"]
            fx = fx[~fx["match_id"].isin(matches["match_id"])]  # drop any already played
            keep = ["Date", "Time", "HomeTeam", "AwayTeam", "Season", "Kickoff", "match_id"]
            matches = pd.concat([matches, fx[keep]], ignore_index=True)

    out = build_features(matches)
    out["played"] = out["FTR"].notna()
    path = ROOT / "data" / "processed" / "features.csv"
    out.to_csv(path, index=False)
    return out


if __name__ == "__main__":
    ds = build_dataset()
    print(f"{len(ds)} rows ({(~ds.played).sum()} unplayed fixtures) -> data/processed/features.csv")
