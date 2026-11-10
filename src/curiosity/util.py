"""Small shared helpers: logging, dates, and title handling."""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta, timezone

_LOGGER_NAME = "curiosity"


def get_logger() -> logging.Logger:
    """Return the shared, configured logger (idempotent)."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger()


def today_utc() -> date:
    """Current UTC date (pipeline is timezone-independent)."""
    return datetime.now(timezone.utc).date()


def latest_available_day(lag_days: int) -> date:
    """The most recent day Wikimedia pageviews are likely to be complete for."""
    return today_utc() - timedelta(days=max(1, lag_days))


def daterange(end: date, days: int) -> tuple[date, date]:
    """Return ``(start, end)`` covering ``days`` days ending at ``end`` inclusive."""
    return end - timedelta(days=days - 1), end


def ymd(d: date) -> str:
    """Format a date as ``YYYYMMDD`` (Wikimedia per-article format)."""
    return d.strftime("%Y%m%d")


def iso(d: date) -> str:
    """Format a date as ``YYYY-MM-DD``."""
    return d.strftime("%Y-%m-%d")


def parse_iso(s: str) -> date:
    """Parse ``YYYY-MM-DD`` back into a date."""
    return datetime.strptime(s, "%Y-%m-%d").date()


def title_to_display(title: str) -> str:
    """Turn a Wikipedia DB title (``Solar_eclipse``) into human text."""
    return title.replace("_", " ").strip()


def display_to_title(name: str) -> str:
    """Inverse of :func:`title_to_display`."""
    return name.replace(" ", "_").strip()


def wiki_url(title: str, project: str = "en.wikipedia") -> str:
    """Build a canonical Wikipedia article URL from a DB title."""
    lang = project.split(".")[0]
    return f"https://{lang}.wikipedia.org/wiki/{title}"
