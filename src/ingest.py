"""Download EPL match data from football-data.co.uk and load it into a clean DataFrame.

Usage:
    python -m src.ingest            # download all seasons + current fixtures
    python -m src.ingest --current  # refresh only the current season (for weekly updates)
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Season codes as used by football-data.co.uk ("1920" = 2019-20).
HISTORICAL_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
CURRENT_SEASON = "2627"
ALL_SEASONS = HISTORICAL_SEASONS + [CURRENT_SEASON]

# Columns we keep. AvgC* = market-average *closing* odds (present in every season;
# Pinnacle closing odds are missing from 2026-27, so we use the average for consistency).
KEEP_COLS = ["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
             "AvgCH", "AvgCD", "AvgCA"]


def season_label(code: str) -> str:
    """'1920' -> '2019-20'."""
    return f"20{code[:2]}-{code[2:]}"


def _get(url: str) -> bytes:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "epl-predictor/1.0"})
    resp.raise_for_status()
    return resp.content


def download_season(code: str, raw_dir: Path = RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"E0_{code}.csv"
    path.write_bytes(_get(BASE_URL.format(code=code)))
    return path


def download_fixtures(raw_dir: Path = RAW_DIR) -> Path:
    """Upcoming (unplayed) EPL fixtures for the current season."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = _read_csv_bytes(_get(FIXTURES_URL))
    df = df[df["Div"] == "E0"]
    path = raw_dir / f"fixtures_{CURRENT_SEASON}.csv"
    df.to_csv(path, index=False)
    return path


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    # Files carry a UTF-8 BOM; older ones occasionally contain latin-1 characters.
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode CSV")


def load_season(code: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = _read_csv_bytes((raw_dir / f"E0_{code}.csv").read_bytes())
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])  # drop blank trailing rows
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df["Season"] = season_label(code)
    return df


def load_matches(seasons: list[str] | None = None, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """All completed matches, sorted chronologically, with a stable match_id."""
    seasons = seasons or ALL_SEASONS
    frames = [load_season(c, raw_dir) for c in seasons if (raw_dir / f"E0_{c}.csv").exists()]
    df = pd.concat(frames, ignore_index=True)
    df["Time"] = df["Time"].fillna("15:00")
    df["Kickoff"] = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"])
    df = df.sort_values(["Kickoff", "HomeTeam"], kind="stable").reset_index(drop=True)
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["match_id"] = (df["Date"].dt.strftime("%Y%m%d") + "_" + df["HomeTeam"] + "_" + df["AwayTeam"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", action="store_true", help="only refresh the current season")
    args = parser.parse_args()

    for code in ([CURRENT_SEASON] if args.current else ALL_SEASONS):
        path = download_season(code)
        n = len(load_season(code))
        print(f"{season_label(code)}: {n:4d} matches -> {path.relative_to(ROOT)}")
    fx = download_fixtures()
    print(f"Upcoming fixtures: {len(pd.read_csv(fx))} -> {fx.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
