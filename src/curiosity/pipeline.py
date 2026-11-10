"""The daily pipeline: ingest -> detect -> cluster -> retrieve -> explain ->
forecast -> publish. Invoked by ``uv run curiosity run`` (optionally ``--dry-run``).

Failure handling is defensive throughout: a single failing source or story is
logged and skipped rather than aborting the whole run.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from .agent import explain_story
from .cluster import cluster_stories
from .config import Config
from .db import Store, make_store
from .detect import detect_spikes
from .embeddings import TASK_PASSAGE, Embedder, make_embedder
from .forecast import forecast_page, log_mae, overall_accuracy
from .ingest import WikiClient, fetch_news
from .llm import LLM, make_llm
from .filters import filter_top_articles
from .models import NewsItem, Story
from .publish import publish_site
from .sampledata import build_sample_inputs
from .util import iso, latest_available_day, log, parse_iso


def _ingest(cfg: Config, latest_day: date):
    """Return ``(history, summaries, news_items, candidate_titles)``.

    In dry-run everything comes from the bundled sample data; otherwise we hit
    the live Wikimedia + RSS sources.
    """
    history_days = int(cfg.wiki.get("history_days", 60))

    if cfg.dry_run:
        top, history, summaries, news = build_sample_inputs(latest_day, history_days)
        top = filter_top_articles(top)  # exercise the real filter path too
        return history, summaries, news, list(history.keys())

    client = WikiClient(cfg)
    top_n = int(cfg.wiki.get("top_n", 1000))
    max_candidates = int(cfg.detect.get("max_candidates", 100))

    raw_top = client.top_articles(latest_day)[:top_n]
    candidates = filter_top_articles(raw_top, limit=max_candidates)
    log.info("Top list: %d raw -> %d candidates after filters.", len(raw_top), len(candidates))

    start = latest_day - timedelta(days=history_days - 1)
    history: dict[str, dict[str, int]] = {}
    for c in candidates:
        title = c["article"]
        series = client.article_history(title, start, latest_day)
        if series:
            # Ensure the latest day reflects the authoritative top-list count.
            series[iso(latest_day)] = c["views"]
            history[title] = series

    # News from RSS feeds (best-effort).
    try:
        news = fetch_news(cfg, latest_day)
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("News ingestion failed entirely: %s", exc)
        news = []

    # Summaries are fetched later, only for detected spikes (fewer requests).
    return history, {}, news, [c["article"] for c in candidates]


def _store_news(store: Store, embedder: Embedder, latest_day: date, news: list[NewsItem]) -> None:
    """Embed news passages and store them (with embeddings) for retrieval."""
    if not news:
        store.replace_news_for_day(iso(latest_day), [])
        return
    texts = [n.text_for_embedding() for n in news]
    try:
        vectors = embedder.embed(texts, TASK_PASSAGE)
        for item, vec in zip(news, vectors):
            item.embedding = vec
    except Exception as exc:
        log.warning("Embedding news failed (%s); storing without vectors.", exc)
    store.replace_news_for_day(iso(latest_day), news)


def _score_previous_day(
    cfg: Config, store: Store, client: WikiClient | None, latest_day: date
) -> list[float]:
    """Score yesterday's forecasts against today's actual views (log MAE)."""
    prev = iso(latest_day - timedelta(days=1))
    scored: list[float] = []
    for fc in store.get_forecasts(prev):
        if fc.actuals and fc.log_mae is not None:
            scored.append(fc.log_mae)
            continue
        # Build actuals for the days the forecast covered (prev+1 .. prev+h).
        actuals: list[int] = []
        base = parse_iso(prev)
        for h in range(1, len(fc.predictions) + 1):
            dday = base + timedelta(days=h)
            if dday > latest_day:
                break
            if client is not None:
                series = client.article_history(fc.topic, dday, dday)
                actuals.append(int(series.get(iso(dday), 0)))
            else:
                series = store.get_pageview_series(fc.topic)
                actuals.append(int(series.get(iso(dday), 0)))
        if actuals:
            mae = log_mae(fc.predictions, actuals)
            store.update_forecast_actuals(prev, fc.topic, actuals, mae)
            scored.append(mae)
    return scored


def _seed_dry_run_accuracy(cfg: Config, store: Store, stories: list[Story], latest_day: date) -> None:
    """In dry-run, store a prior-day forecast WITH actuals so the accuracy
    footer renders a genuine (computed) number rather than a placeholder."""
    if not stories or not stories[0].forecast:
        return
    prev = iso(latest_day - timedelta(days=1))
    fc = stories[0].forecast
    preds = fc.predictions or [fc.history_tail[-1] if fc.history_tail else 1000]
    # Pretend actuals landed ~10% below the prediction.
    actuals = [max(0, int(p * 0.9)) for p in preds]
    mae = log_mae(preds, actuals)
    from .models import Forecast

    seeded = Forecast(
        day=prev,
        topic=stories[0].main_title,
        history_tail=fc.history_tail,
        predictions=preds,
        label=fc.label,
        confidence=fc.confidence,
        actuals=actuals,
        log_mae=mae,
    )
    store.save_forecast(seeded)


def run_pipeline(cfg: Config) -> dict[str, Any]:
    """Execute the full daily pipeline and publish the site. Returns a summary."""
    lag = int(cfg.wiki.get("lag_days", 1))
    latest_day = latest_available_day(lag)
    day_iso = iso(latest_day)
    log.info("=== Curiosity Daily run for %s (dry_run=%s) ===", day_iso, cfg.dry_run)

    store = make_store(cfg)
    embedder = make_embedder(cfg)
    llm: LLM = make_llm(cfg)
    client = None if cfg.dry_run else WikiClient(cfg)

    # 1. Ingest -----------------------------------------------------------
    history, summaries, news, candidates = _ingest(cfg, latest_day)

    # Persist candidate pageview series.
    for title, series in history.items():
        try:
            store.upsert_pageview_series(title, series)
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to store pageviews for %s: %s", title, exc)

    # 2. News: embed + store (needed before retrieval) --------------------
    _store_news(store, embedder, latest_day, news)

    # 3. Detect spikes ----------------------------------------------------
    max_stories = int(cfg.detect.get("max_stories", 15))
    spikes = detect_spikes(
        history,
        day_iso,
        z_threshold=float(cfg.detect.get("z_threshold", 3.5)),
        min_views=int(cfg.detect.get("min_views", 8000)),
        keep=max_stories * 3,
    )
    log.info("Detected %d spiking candidates.", len(spikes))

    # Fetch summaries for the spiking pages (real mode only; dry-run has them).
    if client is not None and spikes:
        for s in spikes:
            if s.title not in summaries:
                try:
                    summaries[s.title] = client.summary_sentence(s.title)
                except Exception:  # pragma: no cover
                    summaries[s.title] = ""

    # 4. Cluster ----------------------------------------------------------
    stories = cluster_stories(spikes, summaries, embedder, cfg, day_iso)
    log.info("Formed %d stories after clustering.", len(stories))

    # 5. Agent explanation + 6. Forecast ---------------------------------
    for story in stories:
        explain_story(story, store, embedder, llm, cfg)
        try:
            series = history.get(story.main_title, {})
            ordered = [series[d] for d in sorted(series.keys())]
            fc = forecast_page(ordered, cfg, day=day_iso, topic=story.main_title)
            story.forecast = fc
            store.save_forecast(fc)
        except Exception as exc:
            log.warning("Forecast failed for %s: %s", story.name, exc)

    # 7. Persist stories --------------------------------------------------
    store.save_stories(day_iso, stories)

    # 8. Score previous day's forecasts ----------------------------------
    try:
        if cfg.dry_run:
            _seed_dry_run_accuracy(cfg, store, stories, latest_day)
        scored = _score_previous_day(cfg, store, client, latest_day)
    except Exception as exc:
        log.warning("Accuracy scoring failed: %s", exc)
        scored = []
    accuracy = overall_accuracy(scored)

    # 9. Sizes + run record ----------------------------------------------
    sizes = store.approx_sizes()
    run_record = {
        "day": day_iso,
        "stories": len(stories),
        "candidates": len(candidates),
        "spikes": len(spikes),
        "news_items": len(news),
        "jina_tokens": embedder.tokens_used,
        "sizes": sizes,
        "accuracy_log_mae": accuracy,
    }
    store.save_run(run_record)
    log.info("Run sizes: %s", sizes)
    log.info("Jina tokens used this run: %d", embedder.tokens_used)

    # 10. Publish ---------------------------------------------------------
    archive_days = int(cfg.site.get("archive_days", 60))
    days = store.list_days()[:archive_days]
    if day_iso not in days:
        days = [day_iso] + days
    stories_by_day = {d: store.get_stories(d) for d in days}
    stories_by_day[day_iso] = stories  # ensure freshest in-memory copy
    publish_site(cfg, day_iso, stories_by_day, sizes=sizes, accuracy=accuracy)

    return run_record
