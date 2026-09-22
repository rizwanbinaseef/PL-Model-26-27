"""Weekly update: ingest new 2026-27 results, score each newly completed gameweek
out-of-sample, log it to data/tracking.csv, then retrain on everything and predict
upcoming fixtures.

For every gameweek not yet in the log:
  1. train each model on all matches that kicked off BEFORE the gameweek started;
  2. compute the gameweek's features with its own results hidden (i.e. exactly what was
     knowable before the first kickoff);
  3. score predictions (accuracy, log loss, RPS) and append a row per model to tracking.csv.
Finally the models are refit on all completed matches and saved as models/*_latest.joblib,
and predictions for known upcoming fixtures are written to data/upcoming_predictions.csv.

Idempotent: re-running without new results changes nothing.

Usage:
    python update_model.py                     # normal weekly run
    python update_model.py --no-download       # use data already in data/raw
    python update_model.py --as-of 2026-10-06  # pretend today is this date
Cron (Mondays + Thursdays 09:00, after weekend / midweek rounds):
    0 9 * * 1,4  cd /path/to/epl-predictor && .venv/bin/python update_model.py >> update.log 2>&1
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, build_dataset, build_features
from src.gameweeks import assign_gameweeks, completed_gameweeks
from src.ingest import CURRENT_SEASON, ROOT, download_fixtures, download_season, season_label
from src.models import MODEL_FACTORIES, bookmaker_proba, evaluate, fit, predict_proba

TRACKING_PATH = ROOT / "data" / "tracking.csv"
PREDICTIONS_PATH = ROOT / "data" / "predictions.csv"
UPCOMING_PATH = ROOT / "data" / "upcoming_predictions.csv"
MODELS_DIR = ROOT / "models"
CURRENT_LABEL = season_label(CURRENT_SEASON)
TRACKING_COLS = ["season", "gameweek", "start_date", "end_date", "n_matches", "model",
                 "accuracy", "log_loss", "rps", "n_train_matches", "logged_at"]


def load_best_params() -> dict:
    path = MODELS_DIR / "evaluation.json"
    return json.loads(path.read_text())["best_params"] if path.exists() else {}


def _read_csv(path, cols=None) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)


def score_gameweek(data: pd.DataFrame, gw_rows: pd.DataFrame, params: dict):
    """Train on matches before the gameweek; predict it using pre-gameweek features."""
    start = gw_rows["Kickoff"].min()
    train = data[data["played"] & (data["Kickoff"] < start)]

    hidden = data.copy()
    mask = hidden["match_id"].isin(gw_rows["match_id"])
    hidden.loc[mask, ["FTHG", "FTAG"]] = np.nan          # hide the gameweek's own results
    feats = build_features(hidden.drop(columns=[*FEATURE_COLUMNS, "target"]))
    test = feats[feats["match_id"].isin(gw_rows["match_id"])].copy()
    test["target"] = test["FTR"].map({"H": 0, "D": 1, "A": 2})

    metrics, preds = {}, []
    for name, (factory, _) in MODEL_FACTORIES.items():
        model = fit(factory(**params.get(name, {})), train)
        proba = predict_proba(model, test)
        metrics[name] = evaluate(test["target"], proba)
        preds.append(_pred_frame(test, name, proba))
    if test[["AvgCH", "AvgCD", "AvgCA"]].notna().all().all():
        proba = bookmaker_proba(test)
        metrics["bookmaker"] = evaluate(test["target"], proba)
        preds.append(_pred_frame(test, "bookmaker", proba))
    return metrics, pd.concat(preds, ignore_index=True), len(train)


def _pred_frame(df: pd.DataFrame, model: str, proba: np.ndarray) -> pd.DataFrame:
    out = df[["match_id", "Kickoff", "HomeTeam", "AwayTeam", "FTR"]].copy()
    out["model"] = model
    out[["p_home", "p_draw", "p_away"]] = proba.round(4)
    out["predicted"] = np.array(["H", "D", "A"])[proba.argmax(axis=1)]
    return out


def run(as_of: pd.Timestamp, download: bool = True) -> pd.DataFrame:
    if download:
        download_season(CURRENT_SEASON)
        download_fixtures()
    data = build_dataset()
    params = load_best_params()

    season = assign_gameweeks(data[data["Season"] == CURRENT_LABEL])
    tracking = _read_csv(TRACKING_PATH, TRACKING_COLS)
    logged = set(tracking.loc[tracking["season"] == CURRENT_LABEL, "gameweek"].astype(int))
    todo = [g for g in completed_gameweeks(season, as_of) if g not in logged]
    print(f"As of {as_of.date()}: {len(logged)} gameweeks already logged, {len(todo)} new: {todo}")

    new_rows, new_preds = [], []
    for gw in todo:
        rows = season[season["gameweek"] == gw]
        metrics, preds, n_train = score_gameweek(data, rows, params)
        preds.insert(0, "gameweek", gw)
        new_preds.append(preds)
        for model, m in metrics.items():
            new_rows.append({
                "season": CURRENT_LABEL, "gameweek": gw,
                "start_date": rows["Kickoff"].min().date(), "end_date": rows["Kickoff"].max().date(),
                "n_matches": m["n"], "model": model, "accuracy": round(m["accuracy"], 4),
                "log_loss": round(m["log_loss"], 4), "rps": round(m["rps"], 4),
                "n_train_matches": n_train, "logged_at": datetime.now().isoformat(timespec="seconds"),
            })
        summary = "  ".join(f"{k}: acc {v['accuracy']:.3f} / ll {v['log_loss']:.3f}" for k, v in metrics.items())
        print(f"  GW{gw:>2} {rows['Kickoff'].min():%d %b}-{rows['Kickoff'].max():%d %b} "
              f"({len(rows)} matches, trained on {n_train})  {summary}")

    if new_rows:
        tracking = pd.concat([tracking, pd.DataFrame(new_rows)], ignore_index=True)
        tracking.to_csv(TRACKING_PATH, index=False)
        preds = pd.concat([_read_csv(PREDICTIONS_PATH), *new_preds], ignore_index=True)
        preds.to_csv(PREDICTIONS_PATH, index=False)

    # Retrain on everything completed and forecast upcoming fixtures.
    played = data[data["played"]]
    upcoming = data[~data["played"]]
    frames = []
    for name, (factory, _) in MODEL_FACTORIES.items():
        model = fit(factory(**params.get(name, {})), played)
        joblib.dump(model, MODELS_DIR / f"{name}_latest.joblib")
        if len(upcoming):
            frames.append(_pred_frame(upcoming, name, predict_proba(model, upcoming)))
    print(f"Retrained on {len(played)} matches -> models/*_latest.joblib")
    if frames:
        pd.concat(frames, ignore_index=True).drop(columns="FTR").to_csv(UPCOMING_PATH, index=False)
        print(f"Predicted {len(upcoming)} upcoming fixtures -> {UPCOMING_PATH.relative_to(ROOT)}")
    else:
        print("No upcoming fixtures published yet.")
    return tracking


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--as-of", type=pd.Timestamp, default=pd.Timestamp.now(), help="treat this date as today")
    p.add_argument("--no-download", action="store_true", help="skip downloading fresh data")
    args = p.parse_args()
    run(args.as_of, download=not args.no_download)


if __name__ == "__main__":
    main()
