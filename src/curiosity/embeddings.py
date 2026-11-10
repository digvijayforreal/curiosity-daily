"""Embeddings via the Jina AI API (model ``jina-embeddings-v3``).

Task types matter for v3 quality, so we expose the three we use:

* ``retrieval.passage``  — when storing documents (news items).
* ``retrieval.query``    — when searching with a query.
* ``separation``         — when clustering (titles/summaries).

We request reduced 512-dimension vectors (Matryoshka) to save storage and to
match the Atlas vector index, batch inputs per request, retry with backoff on
rate limits, and track token usage against a monthly budget.

A :class:`MockEmbedder` provides deterministic, dependency-free vectors so the
whole pipeline runs under ``--dry-run`` with no API key.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Protocol

import requests

from .config import Config
from .util import log

JINA_URL = "https://api.jina.ai/v1/embeddings"

TASK_PASSAGE = "retrieval.passage"
TASK_QUERY = "retrieval.query"
TASK_SEPARATION = "separation"


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class Embedder(Protocol):
    dimensions: int

    def embed(self, texts: list[str], task: str) -> list[list[float]]: ...

    # Token accounting (0 for the mock).
    @property
    def tokens_used(self) -> int: ...


class JinaEmbedder:
    """Real Jina embeddings client with batching, retries and usage tracking."""

    def __init__(self, cfg: Config) -> None:
        if not cfg.jina_api_key:
            raise RuntimeError("JINA_API_KEY is not set; cannot call Jina embeddings API.")
        self.cfg = cfg
        self.model = cfg.embeddings.get("model", "jina-embeddings-v3")
        self.dimensions = int(cfg.embeddings.get("dimensions", 512))
        self.batch_size = int(cfg.embeddings.get("batch_size", 64))
        self.budget = int(cfg.embeddings.get("monthly_token_budget", 1_000_000))
        self._tokens = 0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {cfg.jina_api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    @property
    def tokens_used(self) -> int:
        return self._tokens

    def _post_batch(self, texts: list[str], task: str) -> list[list[float]]:
        payload = {
            "model": self.model,
            "task": task,
            "dimensions": self.dimensions,
            "input": texts,
        }
        delay = 1.0
        for attempt in range(5):
            try:
                resp = self.session.post(JINA_URL, json=payload, timeout=60)
                if resp.status_code == 429:
                    log.warning("Jina rate limited; backing off %.1fs", delay)
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
                self._tokens += int(data.get("usage", {}).get("total_tokens", 0))
                # Responses are ordered by "index"; sort to be safe.
                rows = sorted(data["data"], key=lambda r: r.get("index", 0))
                return [_l2_normalize(r["embedding"]) for r in rows]
            except requests.RequestException as exc:
                if attempt == 4:
                    raise RuntimeError(f"Jina embeddings request failed: {exc}") from exc
                time.sleep(delay)
                delay *= 2
        return []

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.extend(self._post_batch(batch, task))
        self._warn_if_near_budget()
        return out

    def _warn_if_near_budget(self) -> None:
        if self.budget > 0 and self._tokens >= 0.8 * self.budget:
            log.warning(
                "Jina token usage %d is within 80%% of the monthly budget %d.",
                self._tokens,
                self.budget,
            )


class MockEmbedder:
    """Deterministic hashed bag-of-words embeddings for dry-run and tests.

    Shared vocabulary yields higher cosine similarity, so clustering and vector
    search behave meaningfully offline without any network call.
    """

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions
        self._tokens = 0

    @property
    def tokens_used(self) -> int:
        return self._tokens

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = [t for t in text.lower().split() if t.isalnum() or t.isalpha()]
        self._tokens += max(1, len(tokens))
        for tok in tokens:
            h = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self.dimensions
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return _l2_normalize(vec)

    def embed(self, texts: list[str], task: str) -> list[list[float]]:  # noqa: ARG002
        return [self._vector(t) for t in texts]


def make_embedder(cfg: Config) -> Embedder:
    if cfg.dry_run:
        return MockEmbedder(int(cfg.embeddings.get("dimensions", 512)))
    return JinaEmbedder(cfg)
