"""Forecast each story's main page and label it Rising / Peaking / Fading.

The forecaster tries ``statsforecast``'s AutoETS when available and falls back
to a light damped-trend baseline otherwise, so it stays cheap enough for a
GitHub Actions runner and needs no heavy deps during dry-run or tests.

Accuracy is scored the next day: we compare the stored prediction against the
actual views and record a per-day log MAE for the "how accurate were we"
footer.
"""

from __future__ import annotations

import math
from statistics import median

from .config import Config
from .models import Forecast
from .util import log


def label_forecast(
    last_value: float,
    predictions: list[int],
    *,
    rising_pct: float,
    fading_pct: float,
) -> str:
    """Classify the trajectory relative to the latest observed value."""
    if not predictions:
        return "Peaking"
    future = predictions[-1]
    base = max(last_value, 1.0)
    change = (future - base) / base
    if change >= rising_pct:
        return "Rising"
    if change <= fading_pct:
        return "Fading"
    return "Peaking"


def _confidence(history: list[int]) -> str:
    """Rough confidence from history length and volatility (coefficient of var)."""
    if len(history) < 14:
        return "low"
    tail = history[-21:]
    m = sum(tail) / len(tail)
    if m <= 0:
        return "low"
    var = sum((x - m) ** 2 for x in tail) / len(tail)
    cv = math.sqrt(var) / m
    if cv < 0.4:
        return "high"
    if cv < 1.0:
        return "medium"
    return "low"


def _baseline_forecast(history: list[int], horizon: int) -> list[int]:
    """Mean-reversion fallback forecast.

    Pageview spikes overwhelmingly decay back toward the page's own baseline, so
    the baseline model relaxes the current value toward the recent median with a
    geometric decay. A page sitting at its baseline stays roughly flat
    (Peaking); a page far above baseline reverts downward (Fading); a page that
    is still climbing above a rising recent median reads as Rising.
    """
    if not history:
        return [0] * horizon
    current = float(history[-1])
    window = history[-21:-1] if len(history) > 1 else history
    med = float(median(window)) if window else current
    phi = 0.6
    preds: list[int] = []
    for h in range(1, horizon + 1):
        val = med + (current - med) * (phi**h)
        preds.append(max(0, int(round(val))))
    return preds


def _ets_forecast(history: list[int], horizon: int) -> list[int] | None:
    """Try statsforecast AutoETS; return None if unavailable or it errors."""
    if len(history) < 14:
        return None
    try:  # pragma: no cover - exercised only when statsforecast is installed
        import numpy as np
        import pandas as pd
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS

        df = pd.DataFrame(
            {
                "unique_id": "x",
                "ds": pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(history)),
                "y": np.asarray(history, dtype=float),
            }
        )
        sf = StatsForecast(models=[AutoETS(season_length=7)], freq="D", n_jobs=1)
        sf.fit(df)
        fc = sf.predict(h=horizon)
        col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        return [max(0, int(round(v))) for v in fc[col].tolist()]
    except Exception as exc:  # pragma: no cover
        log.debug("AutoETS failed, using baseline: %s", exc)
        return None


def forecast_page(
    history: list[int],
    cfg: Config,
    *,
    day: str,
    topic: str,
) -> Forecast:
    """Build a :class:`Forecast` for one page's view history."""
    horizon = int(cfg.forecast.get("horizon", 3))
    rising = float(cfg.forecast.get("rising_pct", 0.08))
    fading = float(cfg.forecast.get("fading_pct", -0.08))

    preds = _ets_forecast(history, horizon) or _baseline_forecast(history, horizon)
    last_value = float(history[-1]) if history else 0.0
    label = label_forecast(last_value, preds, rising_pct=rising, fading_pct=fading)

    return Forecast(
        day=day,
        topic=topic,
        history_tail=history[-14:],
        predictions=preds,
        label=label,
        confidence=_confidence(history),
    )


def log_mae(predictions: list[int], actuals: list[int]) -> float:
    """Mean absolute error in log space (robust to the scale of view counts)."""
    n = min(len(predictions), len(actuals))
    if n == 0:
        return 0.0
    total = sum(
        abs(math.log1p(max(0, predictions[i])) - math.log1p(max(0, actuals[i])))
        for i in range(n)
    )
    return total / n


def overall_accuracy(mae_values: list[float]) -> float | None:
    """Median log MAE across scored forecasts (None when nothing scored yet)."""
    vals = [m for m in mae_values if m is not None]
    if not vals:
        return None
    return round(median(vals), 3)
