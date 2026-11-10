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
        log.error("Pipeline failed: %s", exc)
        return 1
    log.info("Done. %d stories published for %s.", summary["stories"], summary["day"])
    out = cfg.root / cfg.site.get("output_dir", "site")
    print(f"Site written to: {out}")
    print(f"Open: {out / 'index.html'}")
    return 0


def cmd_setup_indexes(args: argparse.Namespace) -> int:
    from .db import MongoStore

    cfg = load_config(args.config)
    if not cfg.mongodb_uri:
        log.error("MONGODB_URI is not set. Export it or add it to .env first.")
        return 1
    store = MongoStore(cfg)
    created = store.setup_search_indexes()
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

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
