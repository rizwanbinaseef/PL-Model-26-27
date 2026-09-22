"""Tests for metrics and baselines."""
import numpy as np
import pandas as pd
import pytest

from src.models import base_rate_proba, bookmaker_proba, evaluate, rps


def test_rps_perfect_and_worst():
    assert rps([0], np.array([[1.0, 0, 0]])) == 0
    assert rps([2], np.array([[1.0, 0, 0]])) == 1        # confident home, away won


def test_rps_rewards_near_misses():
    # home win: predicting draw is "closer" than predicting away
    assert rps([0], np.array([[0, 1.0, 0]])) < rps([0], np.array([[0, 0, 1.0]]))


def test_bookmaker_proba_removes_overround():
    df = pd.DataFrame({"AvgCH": [2.0], "AvgCD": [3.4], "AvgCA": [3.8]})
    p = bookmaker_proba(df)
    assert p.sum() == pytest.approx(1)
    assert p[0, 0] > p[0, 1] > p[0, 2]   # shorter odds -> higher probability


def test_base_rate_matches_training_frequencies():
    train = pd.DataFrame({"target": [0, 0, 1, 2]})
    assert base_rate_proba(train, 2)[1].tolist() == [0.5, 0.25, 0.25]


def test_evaluate_uses_explicit_predictions():
    proba = np.array([[0.2, 0.3, 0.5]] * 2)
    m = evaluate([0, 0], proba, pred=np.array([0, 0]))
    assert m["accuracy"] == 1 and m["log_loss"] == pytest.approx(-np.log(0.2))
