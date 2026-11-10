"""Tests for the title filters."""

from __future__ import annotations

from curiosity.filters import filter_top_articles, is_junk_title


def test_is_junk_title_blocks_namespaces_and_meta():
    for bad in [
        "Main_Page",
        "Special:Search",
        "Wikipedia:About",
        "File:Example.jpg",
        "Category:Living_people",
        "List_of_tallest_buildings",
        "Index_of_articles",
        "2026",
        "2026_in_film",
    ]:
        assert is_junk_title(bad), bad


def test_is_junk_title_keeps_real_topics():
    for good in [
        "Solar_eclipse",
        "Taylor_Swift",
        "Dune:_Part_Two",       # colon is part of a real title
        "Mission:_Impossible",
        "Nvidia",
    ]:
        assert not is_junk_title(good), good


def test_filter_top_articles_removes_junk_and_keeps_order():
    articles = [
        {"article": "Main_Page", "views": 500000, "rank": 1},
        {"article": "Solar_eclipse", "views": 90000, "rank": 2},
        {"article": "Special:Search", "views": 80000, "rank": 3},
        {"article": "Taylor_Swift", "views": 70000, "rank": 4},
        {"article": "List_of_films", "views": 60000, "rank": 5},
    ]
    result = filter_top_articles(articles)
    titles = [a["article"] for a in result]
    assert titles == ["Solar_eclipse", "Taylor_Swift"]


def test_filter_top_articles_limit():
    articles = [
        {"article": f"Topic_{i}", "views": 10000 - i, "rank": i} for i in range(20)
    ]
    result = filter_top_articles(articles, limit=5)
    assert len(result) == 5


def test_filter_drops_extreme_bot_traffic():
    # One page 100x the median of the pack is treated as bot traffic.
    articles = [{"article": f"Topic_{i}", "views": 1000, "rank": i} for i in range(20)]
    articles.append({"article": "Weird_page", "views": 1000 * 100, "rank": 21})
    result = filter_top_articles(articles)
    titles = [a["article"] for a in result]
    assert "Weird_page" not in titles


def test_filter_keeps_real_colon_title():
    articles = [
        {"article": "Dune:_Part_Two", "views": 50000, "rank": 1},
        {"article": "Portail:Accueil", "views": 50000, "rank": 2},
    ]
    result = filter_top_articles(articles)
    titles = [a["article"] for a in result]
    assert "Dune:_Part_Two" in titles
    assert "Portail:Accueil" not in titles  # namespace-style colon prefix
