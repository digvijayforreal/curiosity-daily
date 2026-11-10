"""Publish the generated briefing: static site, RSS feed and share card.

Everything is plain HTML/CSS (no templating engine) and a Pillow-drawn PNG, so
the output is tiny, fast and trivially hostable on GitHub Pages.
"""

from __future__ import annotations

import html
import shutil
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any

from .config import Config
from .models import Story
from .util import iso, log, parse_iso

_TEMPLATES = Path(__file__).resolve().parent / "templates"


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def _e(text: str) -> str:
    return html.escape(text or "", quote=True)


def _label_class(label: str) -> str:
    return {"Rising": "rising", "Fading": "fading", "Peaking": "peaking"}.get(label, "")


def _header(site: dict[str, Any], base: str) -> str:
    title = _e(site.get("title", "Curiosity Daily"))
    tagline = _e(site.get("tagline", ""))
    return (
        '<header class="site">'
        f'<div><h1><a href="{base}index.html">{title}</a></h1>'
        f'<div class="tagline">{tagline}</div></div>'
        f'<nav><a href="{base}index.html">Today</a>'
        f'<a href="{base}archive.html">Archive</a>'
        f'<a href="{base}feed.xml">RSS</a></nav>'
        "</header>"
    )


def _footer(sizes: dict[str, Any] | None, accuracy: float | None) -> str:
    acc = (
        f"Yesterday's forecast accuracy (median log MAE): <code>{accuracy}</code>. "
        if accuracy is not None
        else "Forecast accuracy appears once there is a next-day actual to score against. "
    )
    size_note = ""
    if sizes:
        total = sizes.get("total_mb")
        if total is not None:
            size_note = f"Database footprint: <code>{total} MB</code> of the 512 MB free tier. "
    return (
        '<footer class="site">'
        f"{acc}{size_note}"
        "Built automatically from Wikipedia pageview spikes and free news feeds. "
        "Explanations are AI-written from retrieved headlines and link out to sources; "
        "no article text is copied."
        "</footer>"
    )


def _page_shell(title: str, body: str, base: str) -> str:
    return (
        "<!doctype html><html lang=\"en\"><head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_e(title)}</title>"
        f'<link rel="stylesheet" href="{base}style.css">'
        f'<link rel="alternate" type="application/rss+xml" title="Curiosity Daily" href="{base}feed.xml">'
        "</head><body><div class=\"wrap\">"
        f"{body}"
        "</div></body></html>"
    )


def _render_story(story: Story, rank: int) -> str:
    badges = [f'<span class="badge views">{story.total_views:,} views</span>']
    if story.pages:
        badges.append(
            f'<span class="badge">+{story.pages[0].pct_change:.0f}% vs baseline</span>'
        )
    if story.forecast:
        lc = _label_class(story.forecast.label)
        badges.append(
            f'<span class="badge {lc}">{_e(story.forecast.label)}'
            f' · {_e(story.forecast.confidence)} confidence</span>'
        )
    badges_html = "".join(badges)

    if story.reason_unclear or not story.explanation:
        expl = '<p class="explanation unclear">Reason unclear — no supporting news found yet.</p>'
    else:
        expl = f'<p class="explanation">{_e(story.explanation)}</p>'

    main_url = story.pages[0].url if story.pages else "#"
    pages_html = "".join(
        f'<li><a href="{_e(p.url)}">{_e(p.display)}</a> '
        f'<span class="v">{p.views:,}</span> views</li>'
        for p in story.pages
    )
    pages_block = f'<ul class="pages">{pages_html}</ul>' if pages_html else ""

    sources_html = "".join(
        f'<li><a href="{_e(ev.get("url", "#"))}">{_e(ev.get("title", ""))}</a>'
        f' — {_e(ev.get("source", ""))}</li>'
        for ev in story.evidence
    )
    sources_block = (
        f'<ul class="sources"><span class="label">Sources</span>{sources_html}</ul>'
        if sources_html
        else ""
    )

    return (
        '<article class="story">'
        f'<h2><span class="rank">{rank}.</span>'
        f'<a href="{_e(main_url)}">{_e(story.name)}</a></h2>'
        f'<div class="badges">{badges_html}</div>'
        f"{expl}{pages_block}{sources_block}"
        "</article>"
    )


def _render_day_body(
    stories: list[Story], day: str, site: dict[str, Any], base: str
) -> str:
    try:
        nice = parse_iso(day).strftime("%A, %B %-d, %Y")
    except (ValueError, TypeError):
        try:
            nice = parse_iso(day).strftime("%A, %B %d, %Y")  # Windows has no %-d
        except Exception:
            nice = day
    meta = f'<div class="daymeta">Briefing for <span class="date">{_e(nice)}</span> · {len(stories)} stories</div>'
    if not stories:
        body = '<div class="empty">No trending stories were detected for this day.</div>'
    else:
        body = "".join(_render_story(s, i) for i, s in enumerate(stories, start=1))
    return meta + body


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def publish_site(
    cfg: Config,
    latest_day: str,
    stories_by_day: dict[str, list[Story]],
    *,
    sizes: dict[str, Any] | None = None,
    accuracy: float | None = None,
) -> Path:
    """Render the whole static site and return the output directory."""
    site = cfg.site
    base = ""  # all internal links are relative to the site root
    out = cfg.root / site.get("output_dir", "site")
    out.mkdir(parents=True, exist_ok=True)
    (out / "day").mkdir(exist_ok=True)
    (out / "cards").mkdir(exist_ok=True)

    # CSS
    shutil.copyfile(_TEMPLATES / "style.css", out / "style.css")
    # Tell GitHub Pages not to run Jekyll over our files.
    (out / ".nojekyll").write_text("", encoding="utf-8")

    header = _header(site, base)
    footer = _footer(sizes, accuracy)

    # Latest day -> index.html
    latest_stories = stories_by_day.get(latest_day, [])
    index_body = header + _render_day_body(latest_stories, latest_day, site, base) + footer
    (out / "index.html").write_text(
        _page_shell(f"{site.get('title','Curiosity Daily')} — {latest_day}", index_body, base),
        encoding="utf-8",
    )

    # Per-day archive pages.
    day_base = "../"
    for day, stories in stories_by_day.items():
        body = (
            _header(site, day_base)
            + _render_day_body(stories, day, site, day_base)
            + _footer(sizes if day == latest_day else None, accuracy if day == latest_day else None)
        )
        (out / "day" / f"{day}.html").write_text(
            _page_shell(f"{site.get('title','Curiosity Daily')} — {day}", body, day_base),
            encoding="utf-8",
        )

    # Archive index.
    days_sorted = sorted(stories_by_day.keys(), reverse=True)
    items = "".join(
        f'<li><a href="day/{_e(d)}.html">{_e(d)}</a>'
        f'<span class="count">{len(stories_by_day[d])} stories</span></li>'
        for d in days_sorted
    )
    archive_body = (
        header
        + '<div class="daymeta">All past briefings</div>'
        + (f'<ul class="archive-list">{items}</ul>' if items else '<div class="empty">No archive yet.</div>')
        + footer
    )
    (out / "archive.html").write_text(
        _page_shell(f"{site.get('title','Curiosity Daily')} — Archive", archive_body, base),
        encoding="utf-8",
    )

    # RSS + share card for the latest day.
    write_feed(cfg, latest_day, latest_stories, out)
    render_card(cfg, latest_day, latest_stories, out / "cards" / f"{latest_day}.png")

    log.info("Published site to %s (%d days).", out, len(stories_by_day))
    return out


def write_feed(cfg: Config, day: str, stories: list[Story], out: Path) -> Path:
    """Write ``feed.xml`` (RSS 2.0) describing the latest day's stories."""
    site = cfg.site
    base_url = (site.get("base_url") or "").rstrip("/")
    now = format_datetime(datetime.now(timezone.utc))
    link = base_url + "/" if base_url else ""

    items_xml = []
    for s in stories:
        guid = _e(s.story_id)
        story_link = s.pages[0].url if s.pages else link
        desc_parts = []
        if s.reason_unclear or not s.explanation:
            desc_parts.append("Reason unclear — no supporting news found yet.")
        else:
            desc_parts.append(s.explanation)
        desc_parts.append(f"{s.total_views:,} views.")
        if s.forecast:
            desc_parts.append(f"Forecast: {s.forecast.label}.")
        description = _e(" ".join(desc_parts))
        items_xml.append(
            "<item>"
            f"<title>{_e(s.name)}</title>"
            f"<link>{_e(story_link)}</link>"
            f'<guid isPermaLink="false">{guid}</guid>'
            f"<description>{description}</description>"
            "</item>"
        )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0"><channel>'
        f"<title>{_e(site.get('title','Curiosity Daily'))}</title>"
        f"<link>{_e(link)}</link>"
        f"<description>{_e(site.get('tagline',''))}</description>"
        f"<lastBuildDate>{now}</lastBuildDate>"
        f"<pubDate>{now}</pubDate>"
        + "".join(items_xml)
        + "</channel></rss>"
    )
    path = out / "feed.xml"
    path.write_text(xml, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Share card (Pillow)
# --------------------------------------------------------------------------- #
def _load_font(size: int):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_card(cfg: Config, day: str, stories: list[Story], path: Path) -> Path:
    """Draw a 1200x630 PNG share card for the day."""
    from PIL import Image, ImageDraw

    W, H = 1200, 630
    img = Image.new("RGB", (W, H), "#fbfbfe")
    draw = ImageDraw.Draw(img)

    accent = "#5b3df5"
    ink = "#15131c"
    muted = "#6b6980"

    # Accent sidebar.
    draw.rectangle([0, 0, 16, H], fill=accent)

    f_small = _load_font(26)
    f_title = _load_font(58)
    f_tag = _load_font(30)
    f_item = _load_font(38)
    f_rank = _load_font(38)

    pad = 64
    draw.text((pad, 54), cfg.site.get("title", "Curiosity Daily").upper(), font=f_title, fill=ink)
    draw.text((pad, 128), cfg.site.get("tagline", ""), font=f_tag, fill=muted)
    draw.text((pad, 176), day, font=f_small, fill=accent)

    y = 240
    top = stories[:5]
    if not top:
        draw.text((pad, y), "No trending stories detected.", font=f_item, fill=muted)
    for i, s in enumerate(top, start=1):
        name = s.name
        if len(name) > 34:
            name = name[:33] + "…"
        draw.text((pad, y), f"{i}", font=f_rank, fill=accent)
        draw.text((pad + 46, y), name, font=f_item, fill=ink)
        label = s.forecast.label if s.forecast else ""
        if label:
            draw.text((W - pad - 170, y + 4), label, font=f_small, fill=muted)
        y += 68

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG")
    return path
