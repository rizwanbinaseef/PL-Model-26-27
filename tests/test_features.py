"""Tests for feature engineering, focused on the absence of data leakage."""
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, EloConfig, build_features, elo_expected


def make_matches(rows):
    """rows: (season, date, home, away, hg, ag). hg/ag may be None for unplayed."""
    df = pd.DataFrame(rows, columns=["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Kickoff"] = pd.to_datetime(df["Date"]) + pd.Timedelta(hours=15)
    df["FTR"] = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
    df.loc[df.FTHG.isna(), "FTR"] = None
    return df


@pytest.fixture
def toy():
    return make_matches([
        ("S1", "2020-01-01", "A", "B", 2, 0),
        ("S1", "2020-01-01", "C", "D", 1, 1),
        ("S1", "2020-01-08", "B", "C", 0, 3),
        ("S1", "2020-01-08", "D", "A", 1, 2),
        ("S1", "2020-01-15", "A", "C", 1, 1),
        ("S1", "2020-01-15", "B", "D", 4, 0),
        ("S1", "2020-01-22", "C", "A", 0, 1),
        ("S1", "2020-01-22", "D", "B", 2, 2),
    ])


@pytest.fixture(scope="module")
def real():
    from src.ingest import RAW_DIR, load_matches
    if not (RAW_DIR / "E0_2425.csv").exists():
        pytest.skip("raw data not downloaded; run `python -m src.ingest`")
    return load_matches(["2324", "2425"])


# ---------- leakage tests ----------

def test_first_match_uses_only_priors(toy):
    f = build_features(toy).iloc[0]
    assert f.home_elo == f.away_elo == 1500
    assert np.isnan(f.home_form_pts) and np.isnan(f.away_form_ga)


def test_changing_a_result_does_not_affect_that_or_earlier_matches(toy):
    base = build_features(toy)
    changed = toy.copy()
    changed.loc[2, ["FTHG", "FTAG"]] = [9, 0]  # alter match index 2 only
    changed.loc[2, "FTR"] = "H"
    alt = build_features(changed)
    # rows 0..2 must be identical (row 2's own features cannot see its result)
    pd.testing.assert_frame_equal(base.loc[:2, FEATURE_COLUMNS], alt.loc[:2, FEATURE_COLUMNS])
    # ...while at least one later row must differ (sanity: the result is used later)
    assert not base.loc[3:, FEATURE_COLUMNS].equals(alt.loc[3:, FEATURE_COLUMNS])


def test_prefix_invariance_on_real_data(real):
    """Features for the first n matches are identical whether or not later matches exist."""
    full = build_features(real)
    n = 400
    prefix = build_features(real.iloc[:n])
    pd.testing.assert_frame_equal(full.loc[: n - 1, FEATURE_COLUMNS], prefix[FEATURE_COLUMNS])


def test_hiding_future_results_does_not_change_features(real):
    """Blanking results after a cutoff leaves every feature row up to that cutoff unchanged."""
    cutoff = 500
    hidden = real.copy()
    hidden.loc[cutoff:, ["FTHG", "FTAG", "FTR"]] = np.nan
    a, b = build_features(real), build_features(hidden)
    pd.testing.assert_frame_equal(a.loc[:cutoff, FEATURE_COLUMNS], b.loc[:cutoff, FEATURE_COLUMNS])


def test_unplayed_fixtures_do_not_update_state(toy):
    with_future = pd.concat([toy, make_matches([("S1", "2020-01-29", "A", "B", None, None)])],
                            ignore_index=True)
    nxt = make_matches([("S1", "2020-02-05", "A", "D", None, None)])
    a = build_features(pd.concat([toy, nxt], ignore_index=True)).iloc[-1]
    b = build_features(pd.concat([with_future, nxt], ignore_index=True)).iloc[-1]
    assert a.home_elo == b.home_elo and a.home_form_pts == b.home_form_pts


# ---------- correctness tests ----------

def test_elo_update_is_zero_sum_and_correct(toy):
    cfg = EloConfig()
    f = build_features(toy, cfg)
    # after A beat B 2-0 at home, A's rating (seen in its 2nd match) rose by the expected delta
    p = elo_expected(1500, 1500, cfg.home_advantage)
    delta = cfg.k * (np.log(3) + 1) * (1 - p)
    a_second = f[(f.HomeTeam == "A") | (f.AwayTeam == "A")].iloc[1]
    assert a_second.away_elo == pytest.approx(1500 + delta)
    b_second = f[(f.HomeTeam == "B") | (f.AwayTeam == "B")].iloc[1]
    assert b_second.home_elo == pytest.approx(1500 - delta)


def test_rolling_form_and_splits(toy):
    f = build_features(toy)
    # A's 3rd match (2020-01-15 v C): previous = W 2-0 (home), W 2-1 (away)
    r = f.iloc[4]
    assert (r.HomeTeam, r.AwayTeam) == ("A", "C")
    assert r.home_form_pts == 3 and r.home_form_gf == 2 and r.home_form_ga == 0.5
    # home-only split for A: just the 2-0 home win
    assert r.home_home_pts == 3 and r.home_home_gf == 2 and r.home_home_ga == 0


def test_form_window_is_last_five(real):
    f = build_features(real)
    team = "Arsenal"
    games = f[(f.HomeTeam == team) | (f.AwayTeam == team)].reset_index(drop=True)
    pts = [3 if (g.HomeTeam == team) == (g.FTR == "H") and g.FTR != "D" else
           (1 if g.FTR == "D" else 0) for g in games.itertuples()]
    i = 10
    expected = np.mean(pts[i - 5:i])
    row = games.iloc[i]
    got = row.home_form_pts if row.HomeTeam == team else row.away_form_pts
    assert got == pytest.approx(expected)


def test_rest_days(toy):
    f = build_features(toy)
    assert f.iloc[0].home_rest_days == 21          # no previous match -> cap
    assert f.iloc[2].home_rest_days == 7           # B played 7 days earlier
    assert f.iloc[2].rest_diff == 0


def test_season_regression_and_promoted_team():
    cfg = EloConfig()
    df = make_matches([
        ("S1", "2020-01-01", "A", "B", 3, 0),
        ("S2", "2020-08-01", "A", "NEW", 1, 1),
    ])
    f = build_features(df, cfg)
    p =elo_expected(1500, 1500, cfg.home_advantage)
    a_end_s1 = 1500 + cfg.k * (np.log(4) + 1) * (1 - p)
    assert f.iloc[1].home_elo == pytest.approx(a_end_s1 + cfg.season_regression * (1500 - a_end_s1))
    assert f.iloc[1].away_elo == cfg.promoted_rating
