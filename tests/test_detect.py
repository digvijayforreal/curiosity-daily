"""Tests for the robust spike detector."""

from __future__ import annotations

from curiosity.detect import detect_spikes, robust_zscore


def _flat_history(value: int, days: int, latest_day: str, latest: int) -> dict[str, int]:
    series = {f"2026-01-{i + 1:02d}": value for i in range(days)}
    series[latest_day] = latest
    return series


def test_robust_zscore_flags_spike_against_flat_baseline():
    baseline = [100.0] * 30
    # A perfectly flat baseline with a higher latest -> infinite modified z.
    assert robust_zscore(baseline, 5000.0) == float("inf")
    # Latest equal to baseline -> no spike.
    assert robust_zscore(baseline, 100.0) == 0.0


def test_robust_zscore_scales_with_mad():
    baseline = [90, 100, 110, 95, 105, 100, 100, 98, 102, 100]
    z = robust_zscore([float(x) for x in baseline], 300.0)
    assert z > 3.5  # clearly anomalous


def test_detect_spikes_requires_min_views():
    latest_day = "2026-02-01"
    history = {
        "Tiny": _flat_history(5, 40, latest_day, 50),  # spikes relatively but tiny absolute
        "Big": _flat_history(1000, 40, latest_day, 40000),
    }
    spikes = detect_spikes(
        history, latest_day, z_threshold=3.5, min_views=8000, keep=10
    )
    titles = [s.title for s in spikes]
    assert "Big" in titles
    assert "Tiny" not in titles  # filtered by min_views


def test_detect_spikes_ignores_stable_pages():
    latest_day = "2026-02-01"
    # A page whose latest value is in line with its noisy baseline.
    series = {f"2026-01-{i + 1:02d}": 10000 + (i % 5) * 50 for i in range(40)}
    series[latest_day] = 10120
    history = {"Stable": series}
    spikes = detect_spikes(
        history, latest_day, z_threshold=3.5, min_views=8000, keep=10
    )
    assert spikes == []


def test_detect_spikes_ranks_by_score_desc():
    latest_day = "2026-02-01"
    history = {
        "A": _flat_history(1000, 40, latest_day, 20000),
        "B": _flat_history(1000, 40, latest_day, 90000),
    }
    spikes = detect_spikes(
        history, latest_day, z_threshold=3.5, min_views=8000, keep=10
    )
    assert [s.title for s in spikes] == ["B", "A"]


def test_detect_spikes_needs_enough_baseline():
    latest_day = "2026-02-01"
    series = {"2026-01-30": 1000, "2026-01-31": 1000, latest_day: 50000}
    spikes = detect_spikes(
        {"Short": series}, latest_day, z_threshold=3.5, min_views=8000, keep=10
    )
    assert spikes == []  # fewer than 7 baseline points
