"""Hybrid retrieval: keyword search + vector search fused with RRF.

We deliberately run the two searches separately and fuse them in Python with
Reciprocal Rank Fusion rather than relying on ``$rankFusion``, which may not be
available on the Atlas free tier.
"""

from __future__ import annotations

from .config import Config
from .embeddings import TASK_QUERY, Embedder
from .models import NewsItem


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse several ranked lists of keys into one.

    RRF score for a key is the sum over lists of ``1 / (k + rank)`` where
    ``rank`` is 1-based. Keys absent from a list simply contribute nothing.
    Returns ``[(key, score), ...]`` sorted by descending score.
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, key in enumerate(lst, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _fuse(store, embedder: Embedder, query: str, per: int, k: int, top_k: int) -> list[NewsItem]:
    """Run keyword + vector search against one store and fuse with RRF."""
    keyword_hits = store.keyword_search(query, per)
    qvec = embedder.embed([query], TASK_QUERY)
    vector_hits = store.vector_search(qvec[0], per) if qvec else []

    by_url: dict[str, NewsItem] = {}
    for item in keyword_hits + vector_hits:
        by_url.setdefault(item.url, item)

    fused = reciprocal_rank_fusion(
        [[i.url for i in keyword_hits], [i.url for i in vector_hits]], k=k
    )
    results: list[NewsItem] = []
    for url, _score in fused[:top_k]:
        if url in by_url:
            results.append(by_url[url])
    return results


def hybrid_search(
    store,
    embedder: Embedder,
    query: str,
    cfg: Config,
    *,
    top_k: int | None = None,
    fallback_store=None,
) -> list[NewsItem]:
    """Run keyword + vector search for ``query`` and return fused top results.

    ``fallback_store`` (optional) is consulted when the primary store returns
    nothing. This matters during the daily run: Atlas indexes freshly-inserted
    news asynchronously, so an in-memory store holding the day's just-fetched
    news guarantees the agent still has evidence to work from. Atlas remains the
    primary backend and powers later queries (MCP server, CLI) once indexed.
    """
    per = int(cfg.retrieval.get("candidates_per_search", 20))
    k = int(cfg.retrieval.get("rrf_k", 60))
    top_k = top_k or int(cfg.retrieval.get("top_k", 6))

    results = _fuse(store, embedder, query, per, k, top_k)
    if not results and fallback_store is not None:
        results = _fuse(fallback_store, embedder, query, per, k, top_k)
    return results
