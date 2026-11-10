"""Group related spiking pages into single stories.

Pages are embedded (title + first-sentence summary) with the ``separation``
task type and merged greedily: starting from the strongest spike, any later
page whose cosine similarity to an existing cluster's lead page exceeds the
configured threshold joins that cluster. Each story is named by its strongest
page (e.g. "Solar eclipse" absorbs "Totality").
"""

from __future__ import annotations

from .config import Config
from .detect import SpikeResult
from .embeddings import TASK_SEPARATION, Embedder, _l2_normalize
from .models import PageFact, Story
from .util import title_to_display, wiki_url


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are already L2-normalised


def cluster_stories(
    spikes: list[SpikeResult],
    summaries: dict[str, str],
    embedder: Embedder,
    cfg: Config,
    day: str,
) -> list[Story]:
    """Return clustered stories (strongest first), capped at ``max_stories``."""
    if not spikes:
        return []

    project = cfg.wiki.get("project", "en.wikipedia")
    threshold = float(cfg.cluster.get("merge_threshold", 0.82))
    max_stories = int(cfg.detect.get("max_stories", 15))

    # Embed "<display>. <summary>" for each spiking page.
    texts = [
        f"{title_to_display(s.title)}. {summaries.get(s.title, '')}".strip()
        for s in spikes
    ]
    vectors = embedder.embed(texts, TASK_SEPARATION)
    vectors = [_l2_normalize(v) for v in vectors]

    # Greedy clustering keyed on each cluster's lead (strongest) page vector.
    clusters: list[dict] = []  # {"lead_vec", "spikes": [SpikeResult]}
    for spike, vec in zip(spikes, vectors):
        placed = False
        for cl in clusters:
            if _cosine(cl["lead_vec"], vec) >= threshold:
                cl["spikes"].append(spike)
                placed = True
                break
        if not placed:
            clusters.append({"lead_vec": vec, "spikes": [spike]})

    stories: list[Story] = []
    for cl in clusters:
        members = cl["spikes"]
        members.sort(key=lambda s: s.score, reverse=True)
        lead = members[0]
        pages = [
            PageFact(
                title=m.title,
                display=title_to_display(m.title),
                url=wiki_url(m.title, project),
                views=m.views,
                baseline=round(m.baseline, 1),
                zscore=round(m.zscore, 2) if m.zscore != float("inf") else 999.0,
                pct_change=round(m.pct_change, 1),
            )
            for m in members
        ]
        story = Story(
            day=day,
            story_id=f"{day}:{lead.title}",
            main_title=lead.title,
            name=title_to_display(lead.title),
            pages=pages,
            total_views=sum(p.views for p in pages),
            spike_score=lead.score,
        )
        stories.append(story)

    stories.sort(key=lambda s: s.spike_score, reverse=True)
    return stories[:max_stories]
