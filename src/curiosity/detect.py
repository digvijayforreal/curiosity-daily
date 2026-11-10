"""Spike detection using a robust (median/MAD) z-score.

Each candidate page is compared against its OWN baseline (its prior history),
not against other pages, so a normally-quiet page that suddenly jumps scores
highly even if its absolute traffic is lower than a perennially popular page.
A minimum absolute view count guards against tiny-number noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median, pstdev

# Constant that makes the MAD a consistent estimator of the standard
# deviation for normally distributed data (1 / 1.4826).
_MAD_SCALE = 0.6745


@dataclass
class SpikeResult:
    title: str
    views: int
    baseline: float  # median of the baseline window
    zscore: float
    pct_change: float  # percent above baseline median

    @property
    def score(self) -> float:
        """Ranking score: combine spike sharpness with traffic magnitude.

        ``zscore`` captures how unusual the jump is; ``log(views)`` gently
        favours stories with real volume so a jump from 5→50 views does not
        outrank a jump from 2000→40000.
        """
        z = self.zscore if math.isfinite(self.zscore) else 50.0
        return z * math.log10(max(self.views, 10))


def robust_zscore(baseline: list[float], latest: float) -> float:
    """Modified z-score of ``latest`` against the ``baseline`` distribution.

    Falls back to a standard-deviation z-score when the MAD is zero, and to a
    large finite value when the baseline is perfectly constant but ``latest``
    exceeds it (a genuine spike out of a flat history).
    """
    if not baseline:
        return 0.0
    med = median(baseline)
    deviations = [abs(x - med) for x in baseline]
    mad = median(deviations)
    if mad > 0:
        return _MAD_SCALE * (latest - med) / mad
    std = pstdev(baseline)
    if std > 0:
        return (latest - med) / std
    # Perfectly flat baseline.
    if latest > med:
        return float("inf")
    return 0.0


def detect_spikes(
    history: dict[str, dict[str, int]],
    latest_day: str,
    *,
    z_threshold: float,
    min_views: int,
    keep: int,
) -> list[SpikeResult]:
    """Rank candidate pages by spike strength.

    Parameters
    ----------
    history:
        ``{title: {YYYY-MM-DD: views}}`` including ``latest_day``.
    latest_day:
        The day we are detecting spikes for.
    z_threshold:
        Minimum robust z-score to qualify.
    min_views:
        Minimum absolute views on ``latest_day`` to qualify.
    keep:
        Maximum number of results to return (highest score first).
    """
    results: list[SpikeResult] = []
    for title, series in history.items():
        latest = int(series.get(latest_day, 0))
        if latest < min_views:
            continue
        baseline = [float(v) for d, v in series.items() if d != latest_day]
        if len(baseline) < 7:  # need a meaningful baseline window
            continue
        z = robust_zscore(baseline, float(latest))
        if not (z >= z_threshold):
            continue
        med = median(baseline) or 0.0
        pct = ((latest - med) / med * 100.0) if med > 0 else float("inf")
        results.append(
            SpikeResult(
                title=title,
                views=latest,
                baseline=med,
                zscore=z,
                pct_change=pct if math.isfinite(pct) else 9999.0,
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:keep]
