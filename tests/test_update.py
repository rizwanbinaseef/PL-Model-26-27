"""Tests for gameweek grouping and the leakage-safety of the weekly update."""
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, build_features
from src.gameweeks import assign_gameweeks, completed_gameweeks


def fixtures(dates, played=True):
    teams = [f"T{i}" for i in range(20)]
    rows = []
    for i, d in enumerate(dates):
        h, a = teams[(2 * i) % 20], teams[(2 * i + 1) % 20]
        rows.append({"Season": "S", "Kickoff": pd.Timestamp(d) + pd.Timedelta(hours=15),
                     "HomeTeam": h, "AwayTeam": a, "played": played,
                     "match_id": f"{d}_{h}_{a}_{i}"})
    return pd.DataFrame(rows)


def test_weekend_is_one_gameweek_and_midweek_another():
    df = fixtures(["2026-10-03"] * 5 + ["2026-10-04"] * 4 + ["2026-10-05"]       # Sat-Mon
                  + ["2026-10-07"] * 5 + ["2026-10-08"] * 5)                     # Wed-Thu
    gw = assign_gameweeks(df)
    assert gw.groupby("gameweek").size().tolist() == [10, 10]


def test_small_block_merges_into_next():
    df = fixtures(["2026-10-01"] + ["2026-10-04"] * 10)   # lone rearranged game, then a round
    gw = assign_gameweeks(df)
    assert gw["gameweek"].nunique() == 1


def test_trailing_small_block_is_not_complete():
    df = fixtures(["2026-10-04"] * 10 + ["2026-10-08"] * 2)
    gw = assign_gameweeks(df)
    assert completed_gameweeks(gw, pd.Timestamp("2026-10-20")) == [1]


def test_gameweek_incomplete_until_all_played_and_day_passed():
    df = pd.concat([fixtures(["2026-10-03"] * 9), fixtures(["2026-10-05"], played=False)])
    gw = assign_gameweeks(df)
    assert completed_gameweeks(gw, pd.Timestamp("2026-10-10")) == []   # Monday game unplayed
    gw["played"] = True
    assert completed_gameweeks(gw, pd.Timestamp("2026-10-05")) == []   # Monday not over yet
    assert completed_gameweeks(gw, pd.Timestamp("2026-10-06")) == [1]


@pytest.fixture(scope="module")
def data():
    from src.ingest import RAW_DIR
    if not (RAW_DIR / "E0_2627.csv").exists():
        pytest.skip("raw data not downloaded; run `python -m src.ingest`")
    from src.features import build_dataset
    return build_dataset()


def test_score_gameweek_uses_only_pre_gameweek_information(data, monkeypatch):
    import update_model
    season = assign_gameweeks(data[data["Season"] == "2026-27"])
    gw_rows = season[season["gameweek"] == 3]
    start = gw_rows["Kickoff"].min()

    seen = {}
    real_fit = update_model.fit

    def spy_fit(model, df):
        seen["max_train_kickoff"] = df["Kickoff"].max()
        return real_fit(model, df)

    real_build = update_model.build_features

    def spy_build(df, *a, **k):
        out = real_build(df, *a, **k)
        seen["features"] = out
        return out

    monkeypatch.setattr(update_model, "fit", spy_fit)
    monkeypatch.setattr(update_model, "build_features", spy_build)
    update_model.score_gameweek(data, gw_rows, {})

    assert seen["max_train_kickoff"] < start
    # The gameweek's features must equal those built from data that ends before the gameweek.
    before = data[data["Kickoff"] < start].drop(columns=[*FEATURE_COLUMNS, "target"])
    blank = gw_rows.drop(columns=[*FEATURE_COLUMNS, "target", "gameweek", "gw_pending_merge"]).copy()
    blank[["FTHG", "FTAG"]] = np.nan
    expected = build_features(pd.concat([before, blank], ignore_index=True))
    got = seen["features"]
    ids = gw_rows["match_id"]
    e = expected[expected.match_id.isin(ids)].set_index("match_id")[FEATURE_COLUMNS].sort_index()
    g = got[got.match_id.isin(ids)].set_index("match_id")[FEATURE_COLUMNS].sort_index()
    pd.testing.assert_frame_equal(e, g)
