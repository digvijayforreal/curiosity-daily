"""Ingestion of free data sources: Wikimedia pageviews and RSS news.

All Wikimedia requests send a descriptive User-Agent as required by the
Wikimedia User-Agent policy (missing/blank UAs are blocked with HTTP 403/429).
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any
from urllib.parse import quote

import requests

from .config import Config
from .models import NewsItem
from .util import iso, log, ymd

REST_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews"
SUMMARY_BASE_TMPL = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"


class WikiClient:
    """Thin HTTP client for the Wikimedia REST pageviews + summary APIs."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": cfg.user_agent(), "Accept": "application/json"}
        )
        self.project = cfg.wiki.get("project", "en.wikipedia")
        self.access = cfg.wiki.get("access", "all-access")
        self.agent = cfg.wiki.get("agent", "all-agents")

    def _get(self, url: str, *, retries: int = 4) -> dict[str, Any] | None:
        """GET with simple exponential backoff. Returns None on 404/final failure."""
        delay = 1.0
        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 404:
                    return None
                if resp.status_code == 429:
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    log.warning("Wikimedia GET failed for %s: %s", url, exc)
                    return None
                time.sleep(delay)
                delay *= 2
        return None

    def top_articles(self, day: date) -> list[dict[str, Any]]:
        """Return the top articles for ``day`` as ``{article, views, rank}`` dicts."""
        url = (
            f"{REST_BASE}/top/{self.project}/{self.access}/"
            f"{day.year}/{day.month:02d}/{day.day:02d}"
        )
        data = self._get(url)
        if not data or "items" not in data or not data["items"]:
            log.warning("No top-articles data for %s", iso(day))
            return []
        articles = data["items"][0].get("articles", [])
        return [
            {
                "article": a.get("article", ""),
                "views": int(a.get("views", 0)),
                "rank": int(a.get("rank", 0)),
            }
            for a in articles
        ]

    def article_history(self, title: str, start: date, end: date) -> dict[str, int]:
        """Return ``{YYYY-MM-DD: views}`` for a single article over a range."""
        enc = quote(title, safe="")
        url = (
            f"{REST_BASE}/per-article/{self.project}/{self.access}/{self.agent}/"
            f"{enc}/daily/{ymd(start)}/{ymd(end)}"
        )
        data = self._get(url)
        if not data or "items" not in data:
            return {}
        out: dict[str, int] = {}
        for item in data["items"]:
            ts = item.get("timestamp", "")  # format: YYYYMMDD00
            if len(ts) >= 8:
                day_iso = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"
                out[day_iso] = int(item.get("views", 0))
        return out

    def summary_sentence(self, title: str) -> str:
        """Return the first-sentence/extract summary of an article (or "")."""
        lang = self.project.split(".")[0]
        url = SUMMARY_BASE_TMPL.format(lang=lang, title=quote(title, safe=""))
        data = self._get(url)
        if not data:
            return ""
        extract = data.get("extract", "") or ""
        # Keep only the first sentence to minimise storage / tokens.
        for sep in (". ", "! ", "? "):
            if sep in extract:
                return extract.split(sep)[0].strip() + sep.strip()
        return extract.strip()

    def resolve_title(self, query: str) -> str | None:
        """Resolve a free-text query to the best matching article DB title.

        Uses the MediaWiki opensearch endpoint (handles misspellings/redirects).
        """
        lang = self.project.split(".")[0]
        url = (
            f"https://{lang}.wikipedia.org/w/api.php?action=opensearch&limit=1"
            f"&namespace=0&format=json&search={quote(query, safe='')}"
        )
        data = self._get(url)
        # opensearch returns [query, [titles], [descriptions], [urls]]
        if isinstance(data, list) and len(data) >= 2 and data[1]:
            return data[1][0].replace(" ", "_")
        return None

    def page_info(self, title: str) -> dict[str, Any]:
        """Return ``{title, display, extract, image, url}`` for an article.

        ``image`` is a Wikipedia/Wikidata thumbnail URL (free to display) or "".
        """
        lang = self.project.split(".")[0]
        url = SUMMARY_BASE_TMPL.format(lang=lang, title=quote(title, safe=""))
        data = self._get(url)
        if not data:
            return {}
        thumb = (data.get("thumbnail") or {}).get("source", "")
        original = (data.get("originalimage") or {}).get("source", "")
        page_url = ((data.get("content_urls") or {}).get("desktop") or {}).get("page", "")
        return {
            "title": data.get("title", title).replace(" ", "_"),
            "display": data.get("title", title),
            "description": data.get("description", "") or "",
            "extract": data.get("extract", "") or "",
            "image": thumb or original,
            "image_large": original or thumb,
            "url": page_url or f"https://{lang}.wikipedia.org/wiki/{quote(title)}",
        }


def _clean_snippet(text: str, limit: int = 280) -> str:
    """Collapse whitespace and strip HTML-ish tags, then truncate.

    We only ever keep a short, our-own summary of third-party text.
    """
    import re

    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text


def fetch_news(cfg: Config, day: date) -> list[NewsItem]:
    """Fetch news items from the configured RSS feeds.

    Only title, source, URL, published time and a short snippet are stored.
    Failures in a single feed are logged and skipped.
    """
    import feedparser

    feeds: list[str] = cfg.news.get("feeds", [])
    per_feed = int(cfg.news.get("max_items_per_feed", 40))
    day_iso = iso(day)
    items: list[NewsItem] = []
    seen_urls: set[str] = set()

    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as exc:  # feedparser is forgiving, but be safe
            log.warning("Failed to parse feed %s: %s", feed_url, exc)
            continue
        source = (parsed.feed.get("title") if parsed.feed else "") or feed_url
        count = 0
        for entry in parsed.entries:
            if count >= per_feed:
                break
            url = entry.get("link", "")
            if not url or url in seen_urls:
                continue
            title = _clean_snippet(entry.get("title", ""), 200)
            if not title:
                continue
            published = ""
            if entry.get("published_parsed"):
                published = time.strftime("%Y-%m-%dT%H:%M:%SZ", entry.published_parsed)
            elif entry.get("published"):
                published = str(entry.get("published"))
            snippet = _clean_snippet(entry.get("summary", "") or entry.get("description", ""))
            items.append(
                NewsItem(
                    title=title,
                    url=url,
                    source=source,
                    published=published,
                    snippet=snippet,
                    fetched_day=day_iso,
                )
            )
            seen_urls.add(url)
            count += 1
    log.info("Fetched %d news items from %d feeds.", len(items), len(feeds))
    return items
