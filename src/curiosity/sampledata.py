"""Loader for the small bundled sample dataset used by ``--dry-run``.

It expands a compact ``spec.json`` into realistic-looking inputs: a top-articles
list, 60 days of per-article history (flat baseline + deterministic noise, with
a spike on the latest day for the trending pages), first-sentence summaries and
a handful of news items. Everything is deterministic so dry-run output is
stable and testable.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

from .models import NewsItem
from .util import iso

_SPEC = Path(__file__).resolve().parent / "sample_data" / "spec.json"


def _load_spec() -> dict:
    with _SPEC.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _series(baseline: int, noise: int, spike: int, days: int, latest: date) -> dict[str, int]:
    """Build a deterministic daily series ending at ``latest`` with a spike."""
    out: dict[str, int] = {}
    for i in range(days):
        d = latest - timedelta(days=(days - 1 - i))
        # Smooth pseudo-random wobble around the baseline (deterministic).
        wobble = int(noise * math.sin(i * 1.3) * 0.5 + noise * math.cos(i * 0.7) * 0.3)
        views = max(0, baseline + wobble)
        out[iso(d)] = views
    out[iso(latest)] = spike  # the spike day
    return out


def build_sample_inputs(latest_day: date, history_days: int = 60):
    """Return ``(top_articles, history, summaries, news_items)`` for dry-run."""
    spec = _load_spec()

    history: dict[str, dict[str, int]] = {}
    summaries: dict[str, str] = {}
    top_articles: list[dict] = []
    rank = 1

    # Junk entries first (high views) so the filters have something to remove.
    for junk in spec["junk_titles"]:
        top_articles.append({"article": junk, "views": 500000 - rank * 1000, "rank": rank})
        rank += 1

    for title, cfg in spec["pages"].items():
        series = _series(
            int(cfg["baseline"]),
            int(cfg.get("noise", 100)),
            int(cfg["spike"]),
            history_days,
            latest_day,
        )
        history[title] = series
        summaries[title] = cfg.get("summary", "")
        top_articles.append({"article": title, "views": series[iso(latest_day)], "rank": rank})
        rank += 1

    # Sort top articles by views descending to mimic the real API ordering.
    top_articles.sort(key=lambda a: a["views"], reverse=True)
    for i, a in enumerate(top_articles, start=1):
        a["rank"] = i

    news_items = [
        NewsItem(
            title=n["title"],
            url=n["url"],
            source=n["source"],
            published=iso(latest_day) + "T08:00:00Z",
            snippet=n["snippet"],
            fetched_day=iso(latest_day),
        )
        for n in spec["news"]
    ]

    return top_articles, history, summaries, news_items
