"""FastAPI backend powering the website's live search / tracking.

Secrets (Tavily, Groq, Jina, Mongo) stay here on the server; the browser only
calls these JSON endpoints. Run locally with ``uv run curiosity serve`` or
deploy to a free host and point the static site at it via CURIOSITY_API_BASE.

Endpoints
---------
GET  /api/health
GET  /api/trending?day=YYYY-MM-DD
GET  /api/track?q=...              live lookup (Wikipedia + image + trend + web + AI)
GET  /api/search?q=...             web + stored-news search
GET  /api/watchlist                tracked subjects with their latest snapshot
POST /api/watch       {"q": ...}   add a subject to the watchlist (+ snapshot now)
DELETE /api/watch?q=...            remove a subject
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .util import iso, latest_available_day, log


def build_app(cfg: Config | None = None):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    cfg = cfg or load_config()

    # Shared clients, created once. The DB is required; others degrade.
    from .db import MongoStore
    from .embeddings import make_embedder
    from .ingest import WikiClient
    from .llm import make_llm
    from .track import track_subject
    from .websearch import make_websearch

    store = MongoStore(cfg)
    wiki = WikiClient(cfg)
    websearch = make_websearch(cfg)
    try:
        embedder = make_embedder(cfg)
    except Exception as exc:  # embeddings optional for the API
        log.warning("Embeddings unavailable: %s", exc)
        embedder = None
    try:
        llm = make_llm(cfg)
    except Exception as exc:
        log.warning("LLM unavailable: %s", exc)
        llm = None

    app = FastAPI(title="Curiosity Daily API", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # public read-only API; safe to allow any origin
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

    @app.get("/api/trending")
    def trending(day: str | None = None) -> dict[str, Any]:
        days = store.list_days()
        day = day or (days[0] if days else None)
        if not day:
            return {"day": None, "stories": []}
        return {"day": day, "stories": [s.public() for s in store.get_stories(day)]}

    @app.get("/api/track")
    def track(q: str) -> dict[str, Any]:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Query 'q' is required.")
        return track_subject(cfg, q, wiki=wiki, websearch=websearch, llm=llm)

    @app.get("/api/search")
    def search(q: str) -> dict[str, Any]:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Query 'q' is required.")
        # Web results (fresh) plus stored-news hybrid results when available.
        web = websearch.search(q, max_results=6, topic="news")
        stored: list[dict[str, Any]] = []
        if embedder is not None:
            from .retrieval import hybrid_search

            for it in hybrid_search(store, embedder, q, cfg):
                stored.append({"title": it.title, "url": it.url, "source": it.source})
        return {
            "query": q,
            "answer": web.answer,
            "web": [
                {"title": it.title, "url": it.url, "snippet": it.snippet, "source": it.source}
                for it in web.items
            ],
            "images": web.images[:6],
            "stored_news": stored,
        }

    def _snapshot(q: str) -> dict[str, Any]:
        """Track a subject now and persist a dated snapshot."""
        result = track_subject(cfg, q, wiki=wiki, websearch=websearch, llm=llm)
        if result.get("found"):
            title = (result.get("title") or q)
            store.add_watch(q, title.replace(" ", "_"), title)
            day = iso(latest_available_day(int(cfg.wiki.get("lag_days", 1))))
            store.save_tracking(
                {
                    "subject": q.strip().lower(),
                    "day": day,
                    "title": title,
                    "image": result.get("image", ""),
                    "summary": result.get("summary", ""),
                    "status": result.get("pageviews", {}).get("status", ""),
                    "today_views": result.get("pageviews", {}).get("today", 0),
                    "web": result.get("web", [])[:4],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        return result

    @app.post("/api/watch")
    def watch(q: str = Body(..., embed=True)) -> dict[str, Any]:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Query 'q' is required.")
        result = _snapshot(q)
        if not result.get("found"):
            raise HTTPException(status_code=404, detail="Subject not found.")
        return {"watching": q, "snapshot": result}

    @app.delete("/api/watch")
    def unwatch(q: str) -> dict[str, Any]:
        store.remove_watch(q)
        return {"removed": q}

    @app.get("/api/watchlist")
    def watchlist() -> dict[str, Any]:
        watched = store.list_watch()
        latest = {t["subject"]: t for t in store.latest_tracking()}
        out = []
        for w in watched:
            snap = latest.get(w["subject"], {})
            out.append({**w, "latest": snap})
        return {"watchlist": out}

    # Optionally serve the static site too (handy for all-in-one local use).
    site_dir = cfg.root / cfg.site.get("output_dir", "site")
    if site_dir.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(site_dir), html=True), name="site")
    else:

        @app.get("/")
        def root() -> JSONResponse:
            return JSONResponse(
                {"service": "Curiosity Daily API", "docs": "/docs", "health": "/api/health"}
            )

    return app


def serve(cfg: Config | None = None, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API with uvicorn."""
    import uvicorn

    uvicorn.run(build_app(cfg), host=host, port=port)
