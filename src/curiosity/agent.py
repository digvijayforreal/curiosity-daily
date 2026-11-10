"""The LLM agent that writes a grounded "why it's trending" per story.

Rules enforced here:

* The agent sees ONLY retrieved news snippets plus the pageview facts.
* If evidence is weak it retries once with a reworded search.
* A verification pass checks every content claim against the snippets; when
  support is insufficient the agent must say "reason unclear" instead of
  guessing.
* We never copy article text — only a short title + link are kept as evidence.
* If the LLM backend is unavailable, the story is published with pageview
  data only and ``reason_unclear`` set.
"""

from __future__ import annotations

import json
import re

from .config import Config
from .embeddings import Embedder
from .llm import LLM, LLMError
from .models import NewsItem, Story
from .retrieval import hybrid_search
from .util import log

_UNCLEAR = "reason unclear"

_STOPWORDS = {
    # articles / connectives / aux verbs
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "after", "that", "this", "is", "are", "was", "were", "has", "have", "had",
    "its", "it", "as", "at", "by", "from", "about", "into", "also", "notes",
    "report", "reports", "reported", "coverage", "attention", "drawing", "says",
    "said", "new", "amid", "over", "more", "they", "their", "been", "being",
    "when", "while", "where", "which", "would", "could", "should", "than", "then",
    "there", "here", "these", "those", "such", "both", "between", "around",
    # generic "why it's trending" narrative vocabulary — these describe the act
    # of trending/reporting, not a specific factual claim, so they must not count
    # for or against grounding.
    "spike", "spiked", "spiking", "surge", "surged", "surging", "rose", "rise",
    "rising", "jump", "jumped", "soared", "climbed", "spotlight", "trending",
    "name", "named", "image", "close", "closely", "drawn", "draw", "drew",
    "ensuing", "prompting", "prompted", "prompt", "launch", "launched",
    "widespread", "media", "public", "online", "activity", "interest",
    "following", "according", "reportedly", "recently", "news", "headlines",
    "story", "stories", "people", "many", "sparked", "sparking", "triggered",
    "triggering", "wake", "renewed", "widely", "amidst", "growing", "wider",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if len(t) > 3 and t not in _STOPWORDS}


def _evidence_vocab(items: list[NewsItem]) -> set[str]:
    vocab: set[str] = set()
    for it in items:
        vocab |= _content_tokens(it.title)
        vocab |= _content_tokens(it.snippet)
    return vocab


def verify_claims(explanation: str, evidence: list[NewsItem], *, min_support: float = 0.4) -> bool:
    """Return True when enough of the explanation's content words are backed by
    the retrieved snippets. Topic words are expected; novel unsupported words
    are what we guard against."""
    if _UNCLEAR in explanation.lower():
        return True
    claim_tokens = _content_tokens(explanation)
    if not claim_tokens:
        return False
    vocab = _evidence_vocab(evidence)
    supported = sum(1 for t in claim_tokens if t in vocab)
    ratio = supported / len(claim_tokens)
    return ratio >= min_support


def _build_prompt(story: Story, evidence: list[NewsItem]) -> str:
    facts = ", ".join(
        f"{p.display} {p.views:,} views (+{p.pct_change:.0f}% vs baseline)"
        for p in story.pages[:4]
    )
    lines = [
        f"[{i}] {it.title} :: {it.snippet} ({it.source})"
        for i, it in enumerate(evidence, start=1)
    ]
    evidence_block = "\n".join(lines) if lines else "(no evidence found)"
    return (
        "You explain why a Wikipedia topic is suddenly trending, using ONLY the "
        "evidence below. Do not use outside knowledge. Do not copy sentences; "
        "summarise in your own words in 2-3 short sentences. If the evidence does "
        'not clearly explain the spike, respond with exactly "reason unclear".\n\n'
        f"TOPIC: {story.name}\n"
        f"PAGEVIEW FACTS: {facts}\n\n"
        "EVIDENCE:\n"
        f"{evidence_block}\n\n"
        'Respond as JSON: {"explanation": "<text or reason unclear>", '
        '"confident": <true|false>}'
    )


def _parse_response(raw: str) -> tuple[str, bool]:
    """Extract (explanation, confident) from the model output, tolerating
    code fences and stray prose around the JSON object."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return str(obj.get("explanation", "")).strip(), bool(obj.get("confident", False))
        except json.JSONDecodeError:
            pass
    # Fall back to treating the whole output as the explanation.
    return text, bool(text) and _UNCLEAR not in text.lower()


def _query_for(story: Story, reworded: bool) -> str:
    terms = [p.display for p in story.pages[:3]]
    base = " ".join(terms)
    return f"{base} latest news why" if reworded else base


def explain_story(
    story: Story,
    store,
    embedder: Embedder,
    llm: LLM,
    cfg: Config,
    *,
    fallback_store=None,
) -> Story:
    """Populate ``story.explanation``, ``story.evidence`` and ``reason_unclear``.

    Degrades gracefully: any failure leaves the story with pageview data only.
    ``fallback_store`` is used for retrieval when the primary (Atlas) store has
    not yet indexed the day's freshly-inserted news.
    """
    try:
        evidence = hybrid_search(
            store, embedder, _query_for(story, False), cfg, fallback_store=fallback_store
        )
        attempts = 0
        explanation = ""
        confident = False

        while attempts < 2:
            if not evidence:
                explanation, confident = _UNCLEAR, False
                break
            prompt = _build_prompt(story, evidence)
            try:
                raw = llm.complete(prompt)
            except LLMError as exc:
                log.warning("LLM unavailable for '%s': %s", story.name, exc)
                story.explanation = ""
                story.reason_unclear = True
                story.evidence = [e.public() for e in evidence[:4]]
                return story

            explanation, confident = _parse_response(raw)
            if confident and _UNCLEAR not in explanation.lower() and verify_claims(explanation, evidence):
                break
            # Weak evidence / unverified: retry once with a reworded search.
            attempts += 1
            if attempts < 2:
                evidence = hybrid_search(
                    store, embedder, _query_for(story, True), cfg, fallback_store=fallback_store
                )

        verified = bool(explanation) and _UNCLEAR not in explanation.lower() and verify_claims(
            explanation, evidence
        )
        if verified:
            story.explanation = explanation
            story.reason_unclear = False
        else:
            story.explanation = ""
            story.reason_unclear = True
        story.evidence = [e.public() for e in evidence[:4]]
    except Exception as exc:  # never let one story kill the run
        log.warning("Agent failed for story '%s': %s", story.name, exc)
        story.explanation = ""
        story.reason_unclear = True
    return story
