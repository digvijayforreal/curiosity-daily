"""Tests for reciprocal rank fusion."""

from __future__ import annotations

from curiosity.retrieval import reciprocal_rank_fusion


def test_rrf_rewards_items_in_both_lists():
    keyword = ["a", "b", "c"]
    vector = ["b", "d", "a"]
    fused = reciprocal_rank_fusion([keyword, vector], k=60)
    ranking = [key for key, _ in fused]
    # "a" (ranks 1 and 3) and "b" (ranks 2 and 1) appear in both -> top two.
    assert set(ranking[:2]) == {"a", "b"}
    assert ranking[0] == "b"  # best combined rank


def test_rrf_handles_disjoint_lists():
    fused = reciprocal_rank_fusion([["x"], ["y"]], k=60)
    scores = dict(fused)
    assert scores["x"] == scores["y"]
    assert len(fused) == 2


def test_rrf_rank_matters():
    # Same single list; earlier rank must score higher.
    fused = reciprocal_rank_fusion([["first", "second", "third"]], k=60)
    scores = dict(fused)
    assert scores["first"] > scores["second"] > scores["third"]


def test_rrf_empty():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_rrf_k_constant_effect():
    # Larger k compresses the differences between ranks.
    small_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    large_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
    gap_small = small_k["a"] - small_k["b"]
    gap_large = large_k["a"] - large_k["b"]
    assert gap_small > gap_large
