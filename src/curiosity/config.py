"""Configuration loading.

Non-secret settings come from ``config.yaml``; secrets come from environment
variables only. ``Config`` is a thin, typed view over the merged result so the
rest of the code never touches ``os.environ`` or raw dicts directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    """Return the repository root (two levels above this file: src/curiosity)."""
    return Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """Typed access to merged YAML config plus environment secrets."""

    raw: dict[str, Any] = field(default_factory=dict)

    # --- secrets / env (never persisted) ---
    mongodb_uri: str = ""
    mongodb_db: str = "curiosity"
    jina_api_key: str = ""
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    groq_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    tavily_api_key: str = ""
    api_base: str = ""  # public backend URL the static site calls
    contact: str = "curiosity-daily (https://github.com/your-user/curiosity-daily)"

    # runtime flags
    dry_run: bool = False
    root: Path = field(default_factory=_project_root)

    # ------------------------------------------------------------------ #
    # Convenient typed accessors for YAML sections.
    # ------------------------------------------------------------------ #
    @property
    def wiki(self) -> dict[str, Any]:
        return self.raw.get("wiki", {})

    @property
    def detect(self) -> dict[str, Any]:
        return self.raw.get("detect", {})

    @property
    def cluster(self) -> dict[str, Any]:
        return self.raw.get("cluster", {})

    @property
    def embeddings(self) -> dict[str, Any]:
        return self.raw.get("embeddings", {})

    @property
    def retrieval(self) -> dict[str, Any]:
        return self.raw.get("retrieval", {})

    @property
    def news(self) -> dict[str, Any]:
        return self.raw.get("news", {})

    @property
    def forecast(self) -> dict[str, Any]:
        return self.raw.get("forecast", {})

    @property
    def site(self) -> dict[str, Any]:
        return self.raw.get("site", {})

    def user_agent(self) -> str:
        """Descriptive User-Agent string required by the Wikimedia API."""
        return f"CuriosityDaily/0.1 {self.contact}"


def load_config(path: str | os.PathLike[str] | None = None, *, dry_run: bool = False) -> Config:
    """Load ``config.yaml`` and overlay environment secrets.

    Parameters
    ----------
    path:
        Optional explicit path to the YAML config. Defaults to ``config.yaml``
        at the project root.
    dry_run:
        When True, external services are mocked by the pipeline.
    """
    root = _project_root()
    cfg_path = Path(path) if path else root / "config.yaml"
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    return Config(
        raw=raw,
        mongodb_uri=os.environ.get("MONGODB_URI", ""),
        mongodb_db=os.environ.get("MONGODB_DB", "curiosity"),
        jina_api_key=os.environ.get("JINA_API_KEY", ""),
        llm_provider=os.environ.get("LLM_PROVIDER", "gemini").lower().strip(),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        groq_api_key=os.environ.get("GROQ_API_KEY", ""),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        tavily_api_key=os.environ.get("TAVILY_API_KEY", ""),
        api_base=os.environ.get("CURIOSITY_API_BASE", ""),
        contact=os.environ.get(
            "CURIOSITY_CONTACT",
            "curiosity-daily (https://github.com/your-user/curiosity-daily)",
        ),
        dry_run=dry_run,
        root=root,
    )
