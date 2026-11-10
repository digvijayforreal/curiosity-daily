"""Tests for the forecast label logic and log MAE."""

from __future__ import annotations

from curiosity.forecast import label_forecast, log_mae


def test_label_rising():
    label = label_forecast(1000, [1100, 1200, 1300], rising_pct=0.08, fading_pct=-0.08)
    assert label == "Rising"


def test_label_fading():
    label = label_forecast(1000, [900, 800, 700], rising_pct=0.08, fading_pct=-0.08)
    assert label == "Fading"


def test_label_peaking_when_flat():
    label = label_forecast(1000, [1010, 1005, 1020], rising_pct=0.08, fading_pct=-0.08)
    assert label == "Peaking"


def test_label_boundary_is_rising():
    # Exactly +8% counts as Rising (>= threshold).
    label = label_forecast(1000, [1000, 1000, 1080], rising_pct=0.08, fading_pct=-0.08)
    assert label == "Rising"


def test_label_empty_predictions_defaults_peaking():
    assert label_forecast(1000, [], rising_pct=0.08, fading_pct=-0.08) == "Peaking"


def test_log_mae_zero_for_perfect_prediction():
    assert log_mae([100, 200, 300], [100, 200, 300]) == 0.0


def test_log_mae_positive_for_error():
    assert log_mae([100], [1000]) > 0.0


def test_log_mae_uses_overlap_only():
    # Extra predictions beyond available actuals are ignored.
    assert log_mae([100, 200, 300], [100]) == 0.0


def test_log_mae_empty():
    assert log_mae([], []) == 0.0
