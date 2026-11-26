"""Focused model-math and contract tests; no research outcomes are synthesized."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.credit_model import (
    PlattCalibrator,
    _ecl_summary,
    _target_metrics,
    _xgb_frame,
    load_contract,
)


def test_ranking_metrics_are_perfect_for_separable_scores():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.05, 0.10, 0.80, 0.95])
    metrics = _target_metrics(labels, scores)
    assert metrics["roc_auc"] == 1.0
    assert metrics["ks"] == 1.0
    assert metrics["positives"] == 2
    assert metrics["actual_rate"] == 0.5


def test_platt_calibrator_is_monotone_and_bounded():
    calibrator = PlattCalibrator(coefficient=1.1, intercept=-0.2)
    probability = np.array([0.001, 0.01, 0.1, 0.5, 0.99])
    calibrated = calibrator.predict(probability)
    assert np.all(np.diff(calibrated) > 0)
    assert np.all((calibrated > 0) & (calibrated < 1))


def test_xgb_frame_preserves_declared_unknown_category():
    frame = pd.DataFrame({
        "score": [720.0, np.nan],
        "state": ["CA", None],
    })
    output = _xgb_frame(
        frame,
        features=["score", "state"],
        numeric=["score"],
        categorical=["state"],
        levels={"state": pd.Index(["CA", "TX"])},
    )
    assert str(output.state.dtype) == "category"
    assert output.loc[1, "state"] == "__MISSING__"


def test_ecl_summary_uses_upb_weighted_rate():
    frame = pd.DataFrame({
        "loan_identifier": ["a", "b"],
        "baseline_upb": [100.0, 300.0],
        "pd": [0.10, 0.20],
        "lgd": [0.50, 0.25],
        "ecl": [5.0, 15.0],
    })
    summary = _ecl_summary(frame, "pd")
    assert summary["expected_loss"] == pytest.approx(20.0)
    assert summary["ecl_rate"] == pytest.approx(0.05)
    assert summary["mean_pd"] == pytest.approx(0.15)


def test_local_modeling_contract_matches_handoff_when_data_are_present():
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/modeling/modeling_contract.json").exists():
        pytest.skip("Portable modeling handoff is not present")
    contract = load_contract(root)
    assert contract["target"] == "serious_delinquency_12m"
    assert len(contract["feature_columns"]) == len(set(contract["feature_columns"]))
    assert "loan_identifier" not in contract["feature_columns"]
    assert "actual_loss" not in contract["feature_columns"]
    assert all(contract["invariants"].values())
