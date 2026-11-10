"""Dataclasses describing the documents that flow through the pipeline.

These map 1:1 onto MongoDB documents (via :meth:`to_doc` / :meth:`from_doc`)
but are plain dataclasses so they are trivial to build, test and serialise to
JSON for the MCP server and the static site.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class NewsItem:
    """A single news article referenced as evidence for a story."""

    title: str
    url: str
    source: str
    published: str  # ISO 8601 string (or "")
    snippet: str  # short, our-own-length-limited summary text
    fetched_day: str = ""  # YYYY-MM-DD the item was ingested
    embedding: list[float] = field(default_factory=list)

    def text_for_embedding(self) -> str:
        return f"{self.title}. {self.snippet}".strip()

    def to_doc(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> "NewsItem":
        return cls(
            title=doc.get("title", ""),
            url=doc.get("url", ""),
            source=doc.get("source", ""),
            published=doc.get("published", ""),
            snippet=doc.get("snippet", ""),
            fetched_day=doc.get("fetched_day", ""),
            embedding=list(doc.get("embedding", [])),
        )

    def public(self) -> dict[str, Any]:
        """JSON-safe view without the (large) embedding."""
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published": self.published,
        }


@dataclass
class PageFact:
    """Pageview facts for one Wikipedia page within a story."""

    title: str
    display: str
    url: str
    views: int
    baseline: float
    zscore: float
    pct_change: float


@dataclass
class Forecast:
    """Forecast for a story's main page."""

    day: str  # the run day this forecast was made (YYYY-MM-DD)
    topic: str  # story id / main page title
    history_tail: list[int] = field(default_factory=list)
    predictions: list[int] = field(default_factory=list)  # next horizon days
    label: str = "Peaking"  # Rising | Peaking | Fading
    confidence: str = "medium"  # low | medium | high
    # Populated the next day when actuals are known.
    actuals: list[int] = field(default_factory=list)
    log_mae: float | None = None

    def to_doc(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> "Forecast":
        return cls(
            day=doc.get("day", ""),
            topic=doc.get("topic", ""),
            history_tail=list(doc.get("history_tail", [])),
            predictions=list(doc.get("predictions", [])),
            label=doc.get("label", "Peaking"),
            confidence=doc.get("confidence", "medium"),
            actuals=list(doc.get("actuals", [])),
            log_mae=doc.get("log_mae"),
        )


@dataclass
class Story:
    """A clustered trending topic for a given day."""

    day: str  # YYYY-MM-DD
    story_id: str  # stable id: f"{day}:{main_title}"
    main_title: str  # Wikipedia DB title of the strongest page
    name: str  # human-readable display name
    pages: list[PageFact] = field(default_factory=list)
    total_views: int = 0
    spike_score: float = 0.0
    explanation: str = ""  # agent-written "why it's trending"
    evidence: list[dict[str, Any]] = field(default_factory=list)  # public news refs
    reason_unclear: bool = False
    forecast: Forecast | None = None

    def to_doc(self) -> dict[str, Any]:
        doc = asdict(self)
        return doc

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> "Story":
        pages = [PageFact(**p) for p in doc.get("pages", [])]
        fc = doc.get("forecast")
        return cls(
            day=doc.get("day", ""),
            story_id=doc.get("story_id", ""),
            main_title=doc.get("main_title", ""),
            name=doc.get("name", ""),
            pages=pages,
            total_views=doc.get("total_views", 0),
            spike_score=doc.get("spike_score", 0.0),
            explanation=doc.get("explanation", ""),
            evidence=list(doc.get("evidence", [])),
            reason_unclear=doc.get("reason_unclear", False),
            forecast=Forecast.from_doc(fc) if fc else None,
        )

    def public(self) -> dict[str, Any]:
        """Clean JSON view for the MCP server and site."""
        return {
            "story_id": self.story_id,
            "day": self.day,
            "name": self.name,
            "main_title": self.main_title,
            "url": self.pages[0].url if self.pages else "",
            "total_views": self.total_views,
            "spike_score": round(self.spike_score, 2),
            "explanation": self.explanation,
            "reason_unclear": self.reason_unclear,
            "pages": [
                {
                    "title": p.display,
                    "url": p.url,
                    "views": p.views,
                    "pct_change": round(p.pct_change, 1),
                    "zscore": round(p.zscore, 2),
                }
                for p in self.pages
            ],
            "evidence": self.evidence,
            "forecast": (
                {
                    "label": self.forecast.label,
                    "confidence": self.forecast.confidence,
                    "predictions": self.forecast.predictions,
                }
                if self.forecast
                else None
            ),
        }
