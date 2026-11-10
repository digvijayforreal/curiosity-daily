"""FastMCP server exposing Curiosity Daily to other AI assistants.

Tools:
  * ``get_trending_today()``      — today's ranked stories.
  * ``why_trending(topic)``       — the grounded explanation + sources.
  * ``forecast_topic(topic)``     — Rising/Peaking/Fading forecast.
  * ``search_news(query)``        — hybrid (keyword + vector) news retrieval.

All tools return clean, JSON-serialisable Python objects. Run with
``uv run curiosity mcp`` (reads MONGODB_URI / JINA_API_KEY from the env).
"""

from __future__ import annotations

from typing import Any

from .config import Config, load_config
from .db import MongoStore
from .embeddings import make_embedder
from .models import Story
from .retrieval import hybrid_search

try:
    from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "fastmcp is not installed. Install deps with `uv sync` before running the MCP server."
    ) from exc


def _latest_day(store: MongoStore) -> str | None:
    days = store.list_days()
    return days[0] if days else None


def _find_story(store: MongoStore, topic: str, *, search_days: int = 7) -> Story | None:
    """Find the best-matching story by topic across the most recent days."""
    needle = topic.strip().lower()
    for day in store.list_days()[:search_days]:
        stories = store.get_stories(day)
        # Exact-ish match first, then substring.
        for s in stories:
            if s.name.lower() == needle or s.main_title.lower() == needle.replace(" ", "_"):
                return s
        for s in stories:
            if needle in s.name.lower() or any(needle in p.display.lower() for p in s.pages):
                return s
    return None


def build_server(cfg: Config | None = None) -> "FastMCP":
    """Construct the FastMCP server with its tools bound to a live store."""
    cfg = cfg or load_config()
    store = MongoStore(cfg)
    embedder = make_embedder(cfg)
    mcp = FastMCP("Curiosity Daily")

    @mcp.tool
    def get_trending_today() -> dict[str, Any]:
        """Return today's trending stories, ranked by spike strength."""
        day = _latest_day(store)
        if not day:
            return {"day": None, "stories": []}
        stories = store.get_stories(day)
        return {"day": day, "stories": [s.public() for s in stories]}

    @mcp.tool
    def why_trending(topic: str) -> dict[str, Any]:
        """Explain why a topic is trending, with sources. Says so if unknown."""
        story = _find_story(store, topic)
        if not story:
            return {"topic": topic, "found": False, "message": "No matching trending story found."}
        return {
            "topic": story.name,
            "found": True,
            "day": story.day,
            "reason_unclear": story.reason_unclear,
            "explanation": story.explanation or "reason unclear",
            "sources": story.evidence,
            "url": story.pages[0].url if story.pages else "",
        }

    @mcp.tool
    def forecast_topic(topic: str) -> dict[str, Any]:
        """Return the view forecast and Rising/Peaking/Fading label for a topic."""
        story = _find_story(store, topic)
        if not story or not story.forecast:
            return {"topic": topic, "found": False, "message": "No forecast available."}
        fc = story.forecast
        return {
            "topic": story.name,
            "found": True,
            "day": story.day,
            "label": fc.label,
            "confidence": fc.confidence,
            "predictions": fc.predictions,
            "history_tail": fc.history_tail,
        }

    @mcp.tool
    def search_news(query: str) -> dict[str, Any]:
        """Hybrid (keyword + vector) search over stored news items."""
        hits = hybrid_search(store, embedder, query, cfg)
        return {"query": query, "results": [h.public() | {"snippet": h.snippet} for h in hits]}

    return mcp


def main() -> None:
    """Entry point for ``curiosity mcp``."""
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
