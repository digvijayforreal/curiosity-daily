"""Web search via the Tavily API, with a deterministic mock for dry-run.

Tavily returns an optional one-line ``answer``, a list of ``results``
(title/url/content) and a list of ``images``. We keep only short snippets and
links — never full article text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import requests

from .config import Config
from .models import NewsItem
from .util import log

TAVILY_URL = "https://api.tavily.com/search"


@dataclass
class WebSearchResult:
    answer: str = ""
    items: list[NewsItem] = field(default_factory=list)
    images: list[str] = field(default_factory=list)


class WebSearch(Protocol):
    def search(self, query: str, *, max_results: int = 6, topic: str = "general") -> WebSearchResult: ...


class TavilyWebSearch:
    """Tavily client. Topic ``news`` biases toward recent news results."""

    def __init__(self, cfg: Config) -> None:
        if not cfg.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is not set; cannot run web search.")
        self.key = cfg.tavily_api_key
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        )

    def search(self, query: str, *, max_results: int = 6, topic: str = "general") -> WebSearchResult:
        payload = {
            "query": query,
            "max_results": max_results,
            "include_answer": True,
            "include_images": True,
            "search_depth": "basic",
            "topic": topic,
        }
        try:
            resp = self.session.post(TAVILY_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            log.warning("Tavily search failed for '%s': %s", query, exc)
            return WebSearchResult()

        def _clip(text: str, n: int = 280) -> str:
            text = (text or "").strip().replace("\n", " ")
            return text[:n] + ("…" if len(text) > n else "")

        items = [
            NewsItem(
                title=_clip(r.get("title", ""), 200),
                url=r.get("url", ""),
                source="web",
                published=r.get("published_date", "") or "",
                snippet=_clip(r.get("content", "")),
            )
            for r in data.get("results", [])
            if r.get("url")
        ]
        images = [u for u in (data.get("images") or []) if isinstance(u, str)]
        return WebSearchResult(answer=_clip(data.get("answer", ""), 400), items=items, images=images)


class MockWebSearch:
    """Offline web search for dry-run/tests."""

    def search(self, query: str, *, max_results: int = 6, topic: str = "general") -> WebSearchResult:  # noqa: ARG002
        return WebSearchResult(
            answer=f"{query} has been in the news recently (sample offline answer).",
            items=[
                NewsItem(
                    title=f"{query}: latest developments",
                    url="https://example.com/news",
                    source="web",
                    published="",
                    snippet=f"A short sample snippet about {query} for offline testing.",
                )
            ],
            images=[],
        )


def make_websearch(cfg: Config) -> WebSearch:
    if cfg.dry_run or not cfg.tavily_api_key:
        return MockWebSearch()
    return TavilyWebSearch(cfg)
