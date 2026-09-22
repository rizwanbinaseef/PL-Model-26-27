"""Train and evaluate models on a time-based split.

    train:      2019-20 .. 2024-25
    validation: 2025-26   (never used for model selection)

Hyper-parameters are chosen on an inner split (train 2019-20..2023-24, tune on 2024-25),
then the chosen configuration is refit on all training seasons.

Usage:  python -m src.train
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.features import build_dataset
from src.ingest import ROOT
from src.models import (MODEL_FACTORIES, base_rate_proba, bookmaker_proba, evaluate, fit,
                        predict_proba, tune)

TRAIN_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
TUNE_SEASON = "2024-25"
VALID_SEASON = "2025-26"
MODELS_DIR = ROOT / "models"


def main() -> None:
    data = build_dataset()
    played = data[data["played"]]
    train = played[played["Season"].isin(TRAIN_SEASONS)]
    valid = played[played["Season"] == VALID_SEASON]
    inner_train = train[train["Season"] != TUNE_SEASON]
    inner_val = train[train["Season"] == TUNE_SEASON]
    print(f"train {len(train)} matches ({TRAIN_SEASONS[0]}..{TRAIN_SEASONS[-1]}), "
          f"validation {len(valid)} matches ({VALID_SEASON})\n")

    results, best_params = {}, {}
    for name, (factory, _) in MODEL_FACTORIES.items():
        params, grid = tune(name, inner_train, inner_val)
        best_params[name] = params
        print(f"[{name}] tuned on {TUNE_SEASON}: best {params} "
              f"(tune log loss {grid.iloc[0]['log_loss']:.4f})")
        model = fit(factory(**params), train)
        results[name] = evaluate(valid["target"], predict_proba(model, valid))
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    y = valid["target"]
    results["baseline_always_home"] = evaluate(
        y, base_rate_proba(train, len(valid)), pred=np.zeros(len(valid), dtype=int))
    results["baseline_bookmaker_closing"] = evaluate(y, bookmaker_proba(valid))

    table = pd.DataFrame(results).T[["n", "accuracy", "log_loss", "rps"]]
    table["n"] = table["n"].astype(int)
    print(f"\nValidation results, {VALID_SEASON} (lower log loss / RPS is better):")
    print(table.round(4).to_string())

    xgb = joblib.load(MODELS_DIR / "xgboost.joblib")
    imp = pd.Series(xgb.feature_importances_, index=xgb.get_booster().feature_names)
    print("\nXGBoost feature importance (gain share), top 8:")
    print(imp.sort_values(ascending=False).head(8).round(3).to_string())

    out = {"train_seasons": TRAIN_SEASONS, "validation_season": VALID_SEASON,
           "best_params": best_params, "results": results}
    (MODELS_DIR / "evaluation.json").write_text(json.dumps(out, indent=2))
    print("\nSaved models/*.joblib and models/evaluation.json")


if __name__ == "__main__":
    main()
