"""Command-line interface: ``curiosity run | setup-indexes | mcp``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import load_config
from .util import log


def _load_dotenv() -> None:
    """Load a local ``.env`` file into the environment if present.

    Minimal parser (no dependency): ``KEY=value`` lines, ``#`` comments, and
    optional surrounding quotes. Existing environment variables win.
    """
    path = Path.cwd() / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    cfg = load_config(args.config, dry_run=args.dry_run)
    try:
        summary = run_pipeline(cfg)
    except Exception as exc:  # top-level guard so CI surfaces a clear error
        msg = str(exc)
        if "SSL handshake failed" in msg or "TLSV1_ALERT_INTERNAL_ERROR" in msg or "ServerSelectionTimeoutError" in type(exc).__name__:
            _explain_mongo_error(exc)
        else:
            log.error("Pipeline failed: %s", exc)
        return 1
    log.info("Done. %d stories published for %s.", summary["stories"], summary["day"])
    out = cfg.root / cfg.site.get("output_dir", "site")
    print(f"Site written to: {out}")
    print(f"Open: {out / 'index.html'}")
    return 0


def _explain_mongo_error(exc: Exception) -> None:
    """Print a concise, human explanation for a MongoDB connection failure."""
    msg = str(exc)
    log.error("Could not connect to MongoDB Atlas.")
    if "TLSV1_ALERT_INTERNAL_ERROR" in msg or "SSL handshake failed" in msg:
        print(
            "\nThe TCP connection reached Atlas but the TLS handshake on port 27017 was\n"
            "rejected. Your Atlas settings are fine - this is almost always the network\n"
            "you are on (office WiFi, a corporate laptop's security software, or an ISP)\n"
            "blocking or inspecting MongoDB's port 27017.\n\n"
            "Fixes, easiest first:\n"
            "  1. Connect to a phone mobile hotspot and run the command again.\n"
            "  2. Turn on a VPN, then run the command again.\n"
            "  3. Skip local DB runs: let GitHub Actions run the pipeline (its cloud\n"
            "     network is not blocked). Locally use `curiosity run --dry-run`.\n"
        )
    elif "timed out" in msg.lower() or "ServerSelectionTimeoutError" in type(exc).__name__:
        print(
            "\nConnection timed out. Check that your current IP is in the Atlas\n"
            "Network Access list (or add 0.0.0.0/0 while testing) and that the\n"
            "cluster is not paused.\n"
        )
    else:
        print(f"\nDetails: {msg[:300]}\n")


def cmd_setup_indexes(args: argparse.Namespace) -> int:
    from .db import MongoStore

    cfg = load_config(args.config)
    if not cfg.mongodb_uri:
        log.error("MONGODB_URI is not set. Export it or add it to .env first.")
        return 1
    try:
        store = MongoStore(cfg)
        created = store.setup_search_indexes()
    except Exception as exc:  # friendly message instead of a raw traceback
        _explain_mongo_error(exc)
        return 1
    if created:
        log.info("Created Atlas search indexes: %s", ", ".join(created))
        print("Created:", ", ".join(created))
    else:
        print("Both Atlas search indexes already exist. Nothing to do.")
    print("Note: Atlas builds search indexes asynchronously; allow a minute before first query.")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import build_server

    cfg = load_config(args.config)
    if not cfg.mongodb_uri:
        log.error("MONGODB_URI is not set; the MCP server needs the database.")
        return 1
    build_server(cfg).run()
    return 0


# --------------------------------------------------------------------------- #
# Read-only query commands (search / browse the stored trending data)
# --------------------------------------------------------------------------- #
def _open_store(args: argparse.Namespace):
    """Open a MongoStore, printing a friendly message on failure. Returns
    ``(cfg, store)`` or ``(cfg, None)``."""
    cfg = load_config(args.config)
    if not cfg.mongodb_uri:
        log.error("MONGODB_URI is not set. Add it to .env first.")
        return cfg, None
    try:
        from .db import MongoStore

        return cfg, MongoStore(cfg)
    except Exception as exc:
        _explain_mongo_error(exc)
        return cfg, None


def _find_story(store, topic: str, *, search_days: int = 14):
    """Best-effort match of a topic to a stored story over recent days."""
    needle = topic.strip().lower()
    for day in store.list_days()[:search_days]:
        stories = store.get_stories(day)
        for s in stories:
            if s.name.lower() == needle or s.main_title.lower() == needle.replace(" ", "_"):
                return s
        for s in stories:
            if needle in s.name.lower() or any(needle in p.display.lower() for p in s.pages):
                return s
    return None


def cmd_trending(args: argparse.Namespace) -> int:
    import json as _json

    cfg, store = _open_store(args)
    if store is None:
        return 1
    day = args.day or (store.list_days()[0] if store.list_days() else None)
    if not day:
        print("No briefings stored yet. Run `curiosity run` first.")
        return 0
    stories = store.get_stories(day)
    if args.json:
        print(_json.dumps({"day": day, "stories": [s.public() for s in stories]}, indent=2))
        return 0
    print(f"\nTrending on {day} — {len(stories)} stories\n")
    for i, s in enumerate(stories, 1):
        label = f" [{s.forecast.label}]" if s.forecast else ""
        print(f"{i:>2}. {s.name}  ({s.total_views:,} views){label}")
        if s.explanation and not s.reason_unclear:
            print(f"    {s.explanation}")
        elif s.reason_unclear:
            print("    (reason unclear - no supporting news found)")
    print()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    cfg, store = _open_store(args)
    if store is None:
        return 1
    from .embeddings import make_embedder
    from .retrieval import hybrid_search

    query = " ".join(args.query)
    embedder = make_embedder(cfg)
    hits = hybrid_search(store, embedder, query, cfg, top_k=args.limit)
    if not hits:
        print(f"No news matched '{query}'. (News items expire after 14 days.)")
        return 0
    print(f"\nTop {len(hits)} news matches for '{query}':\n")
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h.title}\n   {h.source} — {h.url}\n")
    return 0


def cmd_why(args: argparse.Namespace) -> int:
    cfg, store = _open_store(args)
    if store is None:
        return 1
    topic = " ".join(args.topic)
    story = _find_story(store, topic)
    if not story:
        print(f"No trending story found matching '{topic}'. Try `curiosity trending` to see names.")
        return 0
    print(f"\n{story.name}  ({story.day})")
    if story.reason_unclear or not story.explanation:
        print("Why: reason unclear - no supporting news was found.")
    else:
        print(f"Why: {story.explanation}")
    if story.evidence:
        print("Sources:")
        for ev in story.evidence:
            print(f"  - {ev.get('title','')} ({ev.get('source','')}) {ev.get('url','')}")
    print()
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    cfg, store = _open_store(args)
    if store is None:
        return 1
    topic = " ".join(args.topic)
    story = _find_story(store, topic)
    if not story or not story.forecast:
        print(f"No forecast found for '{topic}'. Try `curiosity trending`.")
        return 0
    fc = story.forecast
    print(f"\n{story.name}  ({story.day})")
    print(f"Outlook: {fc.label}  ({fc.confidence} confidence)")
    print(f"Next {len(fc.predictions)} day(s) predicted views: {fc.predictions}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curiosity",
        description="Curiosity Daily — a daily briefing of what the world is suddenly curious about.",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to project root).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the full daily pipeline and publish the site.")
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Use bundled sample data with mocked LLM + embeddings (no API keys, no DB).",
    )
    p_run.set_defaults(func=cmd_run)

    p_idx = sub.add_parser("setup-indexes", help="Create the two Atlas search/vector indexes.")
    p_idx.set_defaults(func=cmd_setup_indexes)

    p_mcp = sub.add_parser("mcp", help="Start the FastMCP server.")
    p_mcp.set_defaults(func=cmd_mcp)

    # --- read-only query commands ---
    p_tr = sub.add_parser("trending", help="List today's (or a given day's) trending stories.")
    p_tr.add_argument("--day", help="Show a specific day (YYYY-MM-DD). Defaults to the latest.")
    p_tr.add_argument("--json", action="store_true", help="Output clean JSON instead of text.")
    p_tr.set_defaults(func=cmd_trending)

    p_se = sub.add_parser("search", help="Hybrid keyword+vector search over stored news.")
    p_se.add_argument("query", nargs="+", help="Words to search for, e.g. curiosity search solar eclipse.")
    p_se.add_argument("--limit", type=int, default=6, help="Max results (default 6).")
    p_se.set_defaults(func=cmd_search)

    p_wy = sub.add_parser("why", help="Show why a topic is trending, with sources.")
    p_wy.add_argument("topic", nargs="+", help="Topic name, e.g. curiosity why jensen huang.")
    p_wy.set_defaults(func=cmd_why)

    p_fc = sub.add_parser("forecast", help="Show the Rising/Peaking/Fading forecast for a topic.")
    p_fc.add_argument("topic", nargs="+", help="Topic name, e.g. curiosity forecast jensen huang.")
    p_fc.set_defaults(func=cmd_forecast)

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
