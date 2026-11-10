"""Filters that remove non-topic pages and obvious noise from the top list.

Kept deliberately pure and dependency-free so it is easy to unit-test.
"""

from __future__ import annotations

import re

# Namespaces / prefixes that are never interesting as "topics".
_BAD_PREFIXES = (
    "Special:",
    "Wikipedia:",
    "File:",
    "Category:",
    "Portal:",
    "Template:",
    "Help:",
    "Talk:",
    "Draft:",
    "Module:",
    "MediaWiki:",
    "User:",
)

_EXACT_BLOCK = {
    "Main_Page",
    "-",
    "",
}

# "List of ...", "Lists of ...", "Index of ..." style meta pages.
_LIST_RE = re.compile(r"^(List|Lists|Index|Outline|Timeline)_of_", re.IGNORECASE)

# Pure-year / date pages ("2026", "2026_in_film") are low-signal.
_YEAR_RE = re.compile(r"^\d{3,4}(_|$)")


def is_junk_title(title: str) -> bool:
    """Return True when a Wikipedia title should be dropped before analysis."""
    if title in _EXACT_BLOCK:
        return True
    if title.startswith(_BAD_PREFIXES):
        return True
    if _LIST_RE.match(title):
        return True
    if _YEAR_RE.match(title):
        return True
    return False


# Namespace-style prefixes ("Foo:") are handled by _BAD_PREFIXES. We do NOT
# drop every title containing a colon, because legitimate articles use them
# (e.g. "Dune:_Part_Two", "Mission:_Impossible").
_NAMESPACE_COLON_RE = re.compile(r"^[A-Za-z][A-Za-z _]{2,30}:")


def looks_like_bot_traffic(title: str, views: int, *, median_top: float) -> bool:
    """Heuristic for obvious automated-traffic pages.

    We flag titles that look like an unknown MediaWiki namespace (a short
    ``Word:`` prefix immediately followed by content, with no space before the
    colon) and pages whose traffic is implausibly far above the median of the
    top list. The median guard keeps this conservative so real viral topics are
    kept.
    """
    # A namespace-style prefix with no space before the colon, e.g. "Portail:".
    # Real article colons are preceded by a word that is itself a normal word
    # and followed by a space in display form; the DB form keeps the underscore
    # ("Dune:_Part_Two"), so a ":_" immediately after the colon is a real title.
    m = _NAMESPACE_COLON_RE.match(title)
    if m and not title[m.end() - 1 :].startswith(":_"):
        # "Foo:Bar" style (namespace) rather than "Foo:_Bar" (article subtitle).
        if ":_" not in title:
            return True
    # Extremely spiky pages far above the pack are usually automated traffic.
    if median_top > 0 and views > median_top * 80:
        return True
    return False


def filter_top_articles(
    articles: list[dict], *, limit: int | None = None
) -> list[dict]:
    """Filter a Wikimedia "top articles" list.

    ``articles`` is a list of ``{"article": str, "views": int, "rank": int}``.
    Returns the cleaned list (optionally truncated to ``limit``).
    """
    cleaned = [a for a in articles if not is_junk_title(a.get("article", ""))]
    if not cleaned:
        return []

    views = sorted(a.get("views", 0) for a in cleaned)
    median_top = views[len(views) // 2] if views else 0.0

    result = [
        a
        for a in cleaned
        if not looks_like_bot_traffic(a["article"], a.get("views", 0), median_top=median_top)
    ]
    if limit is not None:
        result = result[:limit]
    return result
