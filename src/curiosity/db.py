"""Storage layer.

Two interchangeable backends implement the same small surface:

* :class:`MongoStore`  — MongoDB Atlas (M0 free tier), used in production.
* :class:`MemoryStore` — pure-Python, used by ``--dry-run`` and tests so the
  whole pipeline runs with no database and no network.

Both expose identical methods, so the pipeline never cares which is active.
Hybrid retrieval (keyword + vector) is implemented natively on each backend;
the fusion itself lives in :mod:`curiosity.retrieval`.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Protocol

from .config import Config
from .models import Forecast, NewsItem, Story
from .util import log

VECTOR_INDEX = "news_vector_index"
TEXT_INDEX = "news_text_index"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class Store(Protocol):
    """Interface shared by the Mongo and in-memory backends."""

    def upsert_pageview_series(self, title: str, series: dict[str, int]) -> None: ...
    def get_pageview_series(self, title: str) -> dict[str, int]: ...
    def replace_news_for_day(self, day: str, items: list[NewsItem]) -> None: ...
    def keyword_search(self, query: str, limit: int) -> list[NewsItem]: ...
    def vector_search(self, vector: list[float], limit: int) -> list[NewsItem]: ...
    def save_stories(self, day: str, stories: list[Story]) -> None: ...
    def get_stories(self, day: str) -> list[Story]: ...
    def list_days(self) -> list[str]: ...
    def save_forecast(self, fc: Forecast) -> None: ...
    def get_forecasts(self, day: str) -> list[Forecast]: ...
    def update_forecast_actuals(self, day: str, topic: str, actuals: list[int], log_mae: float) -> None: ...
    def save_run(self, run: dict[str, Any]) -> None: ...
    def get_runs(self, limit: int = 30) -> list[dict[str, Any]]: ...
    def approx_sizes(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# In-memory backend
# --------------------------------------------------------------------------- #
class MemoryStore:
    """Dependency-free store used for dry-run and tests."""

    def __init__(self) -> None:
        self._pageviews: dict[str, dict[str, int]] = {}
        self._news: list[NewsItem] = []
        self._stories: dict[str, list[dict[str, Any]]] = {}
        self._forecasts: dict[str, dict[str, dict[str, Any]]] = {}
        self._runs: list[dict[str, Any]] = []
        self._watch: dict[str, dict[str, Any]] = {}
        self._tracking: dict[tuple[str, str], dict[str, Any]] = {}

    def upsert_pageview_series(self, title: str, series: dict[str, int]) -> None:
        self._pageviews.setdefault(title, {}).update(series)

    def get_pageview_series(self, title: str) -> dict[str, int]:
        return dict(self._pageviews.get(title, {}))

    def replace_news_for_day(self, day: str, items: list[NewsItem]) -> None:
        self._news = [n for n in self._news if n.fetched_day != day]
        self._news.extend(items)

    def keyword_search(self, query: str, limit: int) -> list[NewsItem]:
        q = set(_tokens(query))
        if not q:
            return []
        scored: list[tuple[float, NewsItem]] = []
        for item in self._news:
            toks = _tokens(item.title + " " + item.snippet + " " + item.source)
            if not toks:
                continue
            overlap = sum(1 for t in toks if t in q)
            if overlap == 0:
                continue
            # TF-ish score, lightly normalised by length.
            score = overlap / math.sqrt(len(toks))
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:limit]]

    def vector_search(self, vector: list[float], limit: int) -> list[NewsItem]:
        scored = [(_cosine(vector, it.embedding), it) for it in self._news if it.embedding]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:limit]]

    def save_stories(self, day: str, stories: list[Story]) -> None:
        self._stories[day] = [s.to_doc() for s in stories]

    def get_stories(self, day: str) -> list[Story]:
        return [Story.from_doc(d) for d in self._stories.get(day, [])]

    def list_days(self) -> list[str]:
        return sorted(self._stories.keys(), reverse=True)

    def save_forecast(self, fc: Forecast) -> None:
        self._forecasts.setdefault(fc.day, {})[fc.topic] = fc.to_doc()

    def get_forecasts(self, day: str) -> list[Forecast]:
        return [Forecast.from_doc(d) for d in self._forecasts.get(day, {}).values()]

    def update_forecast_actuals(self, day: str, topic: str, actuals: list[int], log_mae: float) -> None:
        bucket = self._forecasts.get(day, {})
        if topic in bucket:
            bucket[topic]["actuals"] = actuals
            bucket[topic]["log_mae"] = log_mae

    def save_run(self, run: dict[str, Any]) -> None:
        self._runs = [r for r in self._runs if r.get("day") != run.get("day")]
        self._runs.append(run)

    def get_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        return sorted(self._runs, key=lambda r: r.get("day", ""), reverse=True)[:limit]

    def approx_sizes(self) -> dict[str, Any]:
        return {
            "pageviews": len(self._pageviews),
            "news_items": len(self._news),
            "stories": sum(len(v) for v in self._stories.values()),
            "forecasts": sum(len(v) for v in self._forecasts.values()),
            "runs": len(self._runs),
        }

    # --- watchlist / tracking ---
    def add_watch(self, subject: str, title: str, display: str) -> None:
        import datetime as _dt

        key = subject.strip().lower()
        self._watch[key] = {
            "subject": key,
            "title": title,
            "display": display,
            "added_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }

    def remove_watch(self, subject: str) -> None:
        self._watch.pop(subject.strip().lower(), None)

    def list_watch(self) -> list[dict[str, Any]]:
        return list(self._watch.values())

    def save_tracking(self, doc: dict[str, Any]) -> None:
        self._tracking[(doc["subject"], doc["day"])] = doc

    def get_tracking(self, subject: str, limit: int = 30) -> list[dict[str, Any]]:
        key = subject.strip().lower()
        rows = [v for (s, _), v in self._tracking.items() if s == key]
        return sorted(rows, key=lambda r: r.get("day", ""), reverse=True)[:limit]

    def latest_tracking(self) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for (s, _), v in self._tracking.items():
            if s not in latest or v.get("day", "") > latest[s].get("day", ""):
                latest[s] = v
        return list(latest.values())


# --------------------------------------------------------------------------- #
# MongoDB backend
# --------------------------------------------------------------------------- #
class MongoStore:
    """MongoDB Atlas backend using pymongo and native Atlas search."""

    def __init__(self, cfg: Config) -> None:
        from pymongo import MongoClient  # lazy import so dry-run needs no driver

        if not cfg.mongodb_uri:
            raise RuntimeError("MONGODB_URI is not set; cannot connect to MongoDB.")
        self.cfg = cfg
        # Use the certifi CA bundle so TLS to Atlas validates consistently across
        # platforms (notably Windows, where the system store can be incomplete).
        client_kwargs: dict[str, Any] = {"appname": "curiosity-daily"}
        try:
            import certifi

            client_kwargs["tlsCAFile"] = certifi.where()
        except ImportError:
            pass
        self.client: Any = MongoClient(cfg.mongodb_uri, **client_kwargs)
        self.db = self.client[cfg.mongodb_db]
        self.pageviews = self.db["pageviews"]
        self.news = self.db["news_items"]
        self.stories = self.db["stories"]
        self.forecasts = self.db["forecasts"]
        self.runs = self.db["runs"]
        self.watchlist = self.db["watchlist"]
        self.tracking = self.db["tracking"]
        self._ensure_basic_indexes()

    def _ensure_basic_indexes(self) -> None:
        """Create normal btree + TTL indexes (idempotent). Search/vector
        indexes are created separately by ``setup-indexes``."""
        from pymongo import ASCENDING

        ttl_days = int(self.cfg.news.get("ttl_days", 14))
        self.pageviews.create_index([("title", ASCENDING)], unique=True)
        self.stories.create_index([("day", ASCENDING)])
        self.stories.create_index([("story_id", ASCENDING)], unique=True)
        self.forecasts.create_index([("day", ASCENDING), ("topic", ASCENDING)], unique=True)
        self.runs.create_index([("day", ASCENDING)], unique=True)
        self.news.create_index([("url", ASCENDING)])
        # TTL index: documents expire ttl_days after `created_at`.
        self.news.create_index("created_at", expireAfterSeconds=ttl_days * 86400)
        self.watchlist.create_index([("subject", ASCENDING)], unique=True)
        self.tracking.create_index([("subject", ASCENDING), ("day", ASCENDING)], unique=True)

    # --- pageviews ---
    def upsert_pageview_series(self, title: str, series: dict[str, int]) -> None:
        existing = self.pageviews.find_one({"title": title}) or {}
        merged = {**existing.get("series", {}), **series}
        self.pageviews.update_one(
            {"title": title},
            {"$set": {"title": title, "series": merged}},
            upsert=True,
        )

    def get_pageview_series(self, title: str) -> dict[str, int]:
        doc = self.pageviews.find_one({"title": title}) or {}
        return dict(doc.get("series", {}))

    # --- news ---
    def replace_news_for_day(self, day: str, items: list[NewsItem]) -> None:
        import datetime as _dt

        self.news.delete_many({"fetched_day": day})
        if not items:
            return
        now = _dt.datetime.now(_dt.timezone.utc)
        docs = []
        for it in items:
            doc = it.to_doc()
            doc["created_at"] = now  # drives the TTL index
            docs.append(doc)
        self.news.insert_many(docs)

    def keyword_search(self, query: str, limit: int) -> list[NewsItem]:
        pipeline = [
            {
                "$search": {
                    "index": TEXT_INDEX,
                    "text": {"query": query, "path": ["title", "snippet", "source"]},
                }
            },
            {"$limit": limit},
            {"$project": {"embedding": 0, "created_at": 0, "_id": 0, "score": {"$meta": "searchScore"}}},
        ]
        try:
            return [NewsItem.from_doc(d) for d in self.news.aggregate(pipeline)]
        except Exception as exc:  # pragma: no cover - needs live Atlas
            log.warning("Atlas keyword search failed (%s); returning empty list.", exc)
            return []

    def vector_search(self, vector: list[float], limit: int) -> list[NewsItem]:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": max(100, limit * 10),
                    "limit": limit,
                }
            },
            {"$project": {"embedding": 0, "created_at": 0, "_id": 0, "score": {"$meta": "vectorSearchScore"}}},
        ]
        try:
            return [NewsItem.from_doc(d) for d in self.news.aggregate(pipeline)]
        except Exception as exc:  # pragma: no cover - needs live Atlas
            log.warning("Atlas vector search failed (%s); returning empty list.", exc)
            return []

    # --- stories ---
    def save_stories(self, day: str, stories: list[Story]) -> None:
        self.stories.delete_many({"day": day})
        if stories:
            self.stories.insert_many([s.to_doc() for s in stories])

    def get_stories(self, day: str) -> list[Story]:
        docs = self.stories.find({"day": day}, {"_id": 0}).sort("spike_score", -1)
        return [Story.from_doc(d) for d in docs]

    def list_days(self) -> list[str]:
        return sorted({d["day"] for d in self.stories.find({}, {"day": 1, "_id": 0})}, reverse=True)

    # --- forecasts ---
    def save_forecast(self, fc: Forecast) -> None:
        self.forecasts.update_one(
            {"day": fc.day, "topic": fc.topic},
            {"$set": fc.to_doc()},
            upsert=True,
        )

    def get_forecasts(self, day: str) -> list[Forecast]:
        return [Forecast.from_doc(d) for d in self.forecasts.find({"day": day}, {"_id": 0})]

    def update_forecast_actuals(self, day: str, topic: str, actuals: list[int], log_mae: float) -> None:
        self.forecasts.update_one(
            {"day": day, "topic": topic},
            {"$set": {"actuals": actuals, "log_mae": log_mae}},
        )

    # --- runs ---
    def save_run(self, run: dict[str, Any]) -> None:
        self.runs.update_one({"day": run["day"]}, {"$set": run}, upsert=True)

    def get_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        return list(self.runs.find({}, {"_id": 0}).sort("day", -1).limit(limit))

    def approx_sizes(self) -> dict[str, Any]:
        """Report approximate collection storage sizes (MB) to watch the 512 MB cap."""
        out: dict[str, Any] = {}
        total = 0.0
        for name in ("pageviews", "news_items", "stories", "forecasts", "runs"):
            try:
                stats = self.db.command("collStats", name)
                size_mb = (stats.get("size", 0) + stats.get("totalIndexSize", 0)) / (1024 * 1024)
            except Exception:
                size_mb = 0.0
            out[name] = round(size_mb, 3)
            total += size_mb
        out["total_mb"] = round(total, 3)
        return out

    # --- watchlist / tracking ---
    def add_watch(self, subject: str, title: str, display: str) -> None:
        import datetime as _dt

        key = subject.strip().lower()
        self.watchlist.update_one(
            {"subject": key},
            {
                "$set": {"subject": key, "title": title, "display": display},
                "$setOnInsert": {"added_at": _dt.datetime.now(_dt.timezone.utc).isoformat()},
            },
            upsert=True,
        )

    def remove_watch(self, subject: str) -> None:
        self.watchlist.delete_one({"subject": subject.strip().lower()})

    def list_watch(self) -> list[dict[str, Any]]:
        return list(self.watchlist.find({}, {"_id": 0}).sort("display", 1))

    def save_tracking(self, doc: dict[str, Any]) -> None:
        self.tracking.update_one(
            {"subject": doc["subject"], "day": doc["day"]}, {"$set": doc}, upsert=True
        )

    def get_tracking(self, subject: str, limit: int = 30) -> list[dict[str, Any]]:
        key = subject.strip().lower()
        return list(
            self.tracking.find({"subject": key}, {"_id": 0}).sort("day", -1).limit(limit)
        )

    def latest_tracking(self) -> list[dict[str, Any]]:
        pipeline = [
            {"$sort": {"day": -1}},
            {"$group": {"_id": "$subject", "doc": {"$first": "$$ROOT"}}},
            {"$replaceRoot": {"newRoot": "$doc"}},
            {"$project": {"_id": 0}},
        ]
        return list(self.tracking.aggregate(pipeline))

    # --- search / vector index management ---
    def setup_search_indexes(self) -> list[str]:
        """Create the two Atlas indexes (vector + text). Idempotent-ish:
        skips creation if an index with the same name already exists."""
        from pymongo.operations import SearchIndexModel

        dims = int(self.cfg.embeddings.get("dimensions", 512))
        created: list[str] = []

        existing = {idx["name"] for idx in self.news.list_search_indexes()}

        if VECTOR_INDEX not in existing:
            vector_model = SearchIndexModel(
                name=VECTOR_INDEX,
                type="vectorSearch",
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": dims,
                            "similarity": "cosine",
                        }
                    ]
                },
            )
            self.news.create_search_index(vector_model)
            created.append(VECTOR_INDEX)

        if TEXT_INDEX not in existing:
            text_model = SearchIndexModel(
                name=TEXT_INDEX,
                type="search",
                definition={
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "title": {"type": "string"},
                            "snippet": {"type": "string"},
                            "source": {"type": "string"},
                        },
                    }
                },
            )
            self.news.create_search_index(text_model)
            created.append(TEXT_INDEX)

        return created


def make_store(cfg: Config) -> Store:
    """Return the appropriate store for the current mode."""
    if cfg.dry_run:
        return MemoryStore()
    return MongoStore(cfg)
