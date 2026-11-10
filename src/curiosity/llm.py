"""A tiny LLM interface with Gemini and Groq backends (both free tier).

The provider is chosen by the ``LLM_PROVIDER`` env var. A :class:`MockLLM`
drives ``--dry-run``: it reads the EVIDENCE lines embedded in the prompt and
returns a grounded JSON answer, so the agent + verification path runs offline.
"""

from __future__ import annotations

import json
import re
import time
from typing import Protocol

import requests

from .config import Config
from .util import log

GEMINI_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(RuntimeError):
    """Raised when the LLM backend is unavailable; callers degrade gracefully."""


class LLM(Protocol):
    def complete(self, prompt: str, *, temperature: float = 0.2) -> str: ...


class GeminiLLM:
    def __init__(self, cfg: Config) -> None:
        if not cfg.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set.")
        self.key = cfg.gemini_api_key
        self.model = cfg.gemini_model
        self.session = requests.Session()

    def complete(self, prompt: str, *, temperature: float = 0.2) -> str:
        url = GEMINI_URL_TMPL.format(model=self.model, key=self.key)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 512},
        }
        delay = 1.0
        for attempt in range(4):
            try:
                resp = self.session.post(url, json=payload, timeout=60)
                if resp.status_code == 429:
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except requests.RequestException as exc:
                if attempt == 3:
                    raise LLMError(f"Gemini request failed: {exc}") from exc
                time.sleep(delay)
                delay *= 2
        raise LLMError("Gemini request failed after retries.")


class GroqLLM:
    def __init__(self, cfg: Config) -> None:
        if not cfg.groq_api_key:
            raise LLMError("GROQ_API_KEY is not set.")
        self.key = cfg.groq_api_key
        self.model = cfg.groq_model
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.key}"})

    def complete(self, prompt: str, *, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 512,
        }
        delay = 1.0
        for attempt in range(4):
            try:
                resp = self.session.post(GROQ_URL, json=payload, timeout=60)
                if resp.status_code == 429:
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except requests.RequestException as exc:
                if attempt == 3:
                    raise LLMError(f"Groq request failed: {exc}") from exc
                time.sleep(delay)
                delay *= 2
        raise LLMError("Groq request failed after retries.")


class MockLLM:
    """Offline LLM. Produces grounded JSON from EVIDENCE lines in the prompt."""

    _EVIDENCE_RE = re.compile(r"^\[\d+\]\s*(.+?)\s*::", re.MULTILINE)

    def complete(self, prompt: str, *, temperature: float = 0.2) -> str:  # noqa: ARG002
        titles = self._EVIDENCE_RE.findall(prompt)
        topic_match = re.search(r"TOPIC:\s*(.+)", prompt)
        topic = topic_match.group(1).strip() if topic_match else "This topic"
        if not titles:
            return json.dumps({"explanation": "reason unclear", "confident": False})
        # Build a 2-sentence explanation strictly from evidence titles so the
        # downstream verification pass finds support for every claim.
        first = titles[0].rstrip(".")
        explanation = f"{topic} is drawing attention after reports that {first}."
        if len(titles) > 1:
            second = titles[1].rstrip(".")
            explanation += f" Coverage also notes that {second}."
        return json.dumps({"explanation": explanation, "confident": True})


def make_llm(cfg: Config) -> LLM:
    if cfg.dry_run:
        return MockLLM()
    provider = cfg.llm_provider
    if provider == "gemini":
        return GeminiLLM(cfg)
    if provider == "groq":
        return GroqLLM(cfg)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use 'gemini' or 'groq'.")
