"""Model definitions, baselines and evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import FEATURE_COLUMNS

CLASSES = [0, 1, 2]  # H, D, A


# ---------------- models ----------------

def make_logreg(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])


def make_xgb(**params) -> XGBClassifier:
    defaults = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
    )
    defaults.update(params)
    return XGBClassifier(**defaults)  # handles NaN natively


LOGREG_GRID = [{"C": c} for c in (0.01, 0.03, 0.1, 0.3, 1.0)]
XGB_GRID = [
    {"max_depth": d, "n_estimators": n, "learning_rate": lr}
    for d in (2, 3, 4) for n in (150, 300, 600) for lr in (0.02, 0.05)
]
MODEL_FACTORIES = {"logreg": (make_logreg, LOGREG_GRID), "xgboost": (make_xgb, XGB_GRID)}


def fit(model, df: pd.DataFrame):
    return model.fit(df[FEATURE_COLUMNS], df["target"].astype(int))


def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURE_COLUMNS])


def tune(name: str, train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Pick hyper-parameters by validation log loss. Returns (best_params, grid results)."""
    factory, grid = MODEL_FACTORIES[name]
    results = []
    for params in grid:
        m = fit(factory(**params), train)
        results.append({**params, **evaluate(val["target"], predict_proba(m, val))})
    res = pd.DataFrame(results).sort_values("log_loss")
    best = {k: res.iloc[0][k] for k in grid[0]}
    best = {k: (int(v) if float(v).is_integer() and k != "C" else float(v)) for k, v in best.items()}
    return best, res


# ---------------- baselines ----------------

def base_rate_proba(train: pd.DataFrame, n: int) -> np.ndarray:
    """Constant prediction = training class frequencies (argmax is always 'home win')."""
    freq = train["target"].value_counts(normalize=True).reindex(CLASSES, fill_value=0).to_numpy()
    return np.tile(freq, (n, 1))


def bookmaker_proba(df: pd.DataFrame, cols=("AvgCH", "AvgCD", "AvgCA")) -> np.ndarray:
    """Closing-odds implied probabilities, overround removed by proportional normalisation."""
    inv = 1.0 / df[list(cols)].to_numpy(dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


# ---------------- metrics ----------------

def rps(y_true, proba: np.ndarray) -> float:
    """Ranked probability score for ordered outcomes H < D < A (lower is better)."""
    y = np.asarray(y_true, dtype=int)
    onehot = np.eye(3)[y]
    cum_diff = np.cumsum(proba, axis=1) - np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum(cum_diff[:, :2] ** 2, axis=1) / 2))


def evaluate(y_true, proba: np.ndarray, pred: np.ndarray | None = None) -> dict:
    y = np.asarray(y_true, dtype=int)
    pred = proba.argmax(axis=1) if pred is None else pred
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-15, 1), labels=CLASSES)),
        "rps": rps(y, proba),
    }
