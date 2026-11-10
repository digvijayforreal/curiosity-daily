"""On-demand tracking of any subject (person, place, topic).

Given a free-text query this resolves a Wikipedia article, pulls its image and
60-day pageview trend (flagging a spike), runs a Tavily web search for recent
news, and writes a short grounded AI summary from the retrieved snippets. The
result is a clean JSON-serialisable dict used by the API and the website.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from .agent import verify_claims
from .config import Config
from .detect import robust_zscore
from .ingest import WikiClient
from .llm import LLM, LLMError, make_llm
from .models import NewsItem
from .util import iso, latest_available_day, log
from .websearch import WebSearch, make_websearch


def _status_from_series(series: dict[str, int], latest_day: str) -> tuple[int, float, str]:
    """Return (today_views, zscore, human status) from a daily series."""
    if not series:
        return 0, 0.0, "no data"
    days = sorted(series.keys())
    latest = int(series.get(latest_day, series[days[-1]]))
    baseline = [float(series[d]) for d in days if d != latest_day]
    if len(baseline) < 7:
        return latest, 0.0, "limited history"
    z = robust_zscore(baseline, float(latest))
    recent = [series[d] for d in days[-4:]]
    rising = len(recent) >= 2 and recent[-1] > recent[0] * 1.1
    falling = len(recent) >= 2 and recent[-1] < recent[0] * 0.9
    if z >= 3.5:
        status = "Spiking now"
    elif rising:
        status = "Rising"
    elif falling:
        status = "Cooling off"
    else:
        status = "Steady"
    return latest, round(z if z != float("inf") else 999.0, 2), status


def _summarize(llm: LLM, subject: str, answer: str, items: list[NewsItem]) -> tuple[str, bool]:
    """Write a short grounded summary. Falls back to Tavily's answer or "" on
    failure. Returns (text, grounded)."""
    if not items and not answer:
        return "", False
    lines = [f"[{i}] {it.title} :: {it.snippet} ({it.url})" for i, it in enumerate(items, 1)]
    prompt = (
        "Summarise, in 2-3 short sentences and in your own words, the latest on the "
        "subject below using ONLY the evidence. Do not copy sentences. If the evidence "
        'is thin, keep it brief and factual.\n\n'
        f"SUBJECT: {subject}\n\nEVIDENCE:\n" + "\n".join(lines) + "\n\nSummary:"
    )
    try:
        text = llm.complete(prompt).strip()
    except LLMError as exc:
        log.warning("LLM summary unavailable for '%s': %s", subject, exc)
        return answer, bool(answer)
    # Prefer the model summary when it is grounded; otherwise use Tavily's answer.
    if text and verify_claims(text, items, min_support=0.3):
        return text, True
    return (answer or text), bool(answer or text)


def track_subject(
    cfg: Config,
    query: str,
    *,
    wiki: WikiClient | None = None,
    websearch: WebSearch | None = None,
    llm: LLM | None = None,
) -> dict[str, Any]:
    """Look up ``query`` and return a structured tracking result."""
    query = (query or "").strip()
    if not query:
        return {"query": query, "found": False, "message": "Empty query."}

    wiki = wiki or WikiClient(cfg)
    websearch = websearch or make_websearch(cfg)
    llm = llm or make_llm(cfg)

    # 1. Resolve the Wikipedia article (handles spelling/redirects).
    title = wiki.resolve_title(query)
    info: dict[str, Any] = wiki.page_info(title) if title else {}

    # 2. Pageview trend + spike status (needs a resolved title).
    pageviews: dict[str, Any] = {}
    if title:
        latest = latest_available_day(int(cfg.wiki.get("lag_days", 1)))
        start = latest - timedelta(days=int(cfg.wiki.get("history_days", 60)) - 1)
        series = wiki.article_history(title, start, latest)
        if series:
            today, z, status = _status_from_series(series, iso(latest))
            ordered_days = sorted(series.keys())
            pageviews = {
                "days": ordered_days[-30:],
                "views": [series[d] for d in ordered_days[-30:]],
                "today": today,
                "zscore": z,
                "status": status,
            }

    # 3. Web search (news-biased) for recent coverage + images.
    web = websearch.search(f"{query} latest news", max_results=6, topic="news")

    # 4. Grounded AI summary.
    subject = info.get("display", query)
    summary, grounded = _summarize(llm, subject, web.answer, web.items)

    # Wikipedia image first, then any Tavily images as extras.
    images: list[str] = []
    if info.get("image"):
        images.append(info["image"])
    images.extend(u for u in web.images if u not in images)

    found = bool(title or web.items)
    return {
        "query": query,
        "found": found,
        "title": info.get("display", query),
        "wikipedia_url": info.get("url", ""),
        "description": info.get("description", ""),
        "extract": info.get("extract", ""),
        "image": info.get("image", ""),
        "images": images[:6],
        "pageviews": pageviews,
        "summary": summary,
        "summary_grounded": grounded,
        "web": [
            {"title": it.title, "url": it.url, "snippet": it.snippet, "source": it.source}
            for it in web.items
        ],
    }
