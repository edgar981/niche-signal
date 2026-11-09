"""Scoring with known inputs, the reachability cap, and merges/reviews applied from YAML."""

import math

import pytest
from helpers import seed_cluster, seed_pain

from niche_miner import db
from niche_miner.score import (
    ClusterStats,
    compute_score,
    compute_stats,
    formula_markdown,
    resolve_merges,
    score_latest,
)
from niche_miner.settings import ScoreWeights

W = ScoreWeights()


def member(i: int, *, author: str | None = None, pain_type: str = "missing_capability",
           excluded: int = 0, who_type: str = "unknown", author_type: str = "unknown",
           spend: str | None = None) -> dict:
    return {"signal_id": i, "author_hash": author or f"a{i}", "pain_type": pain_type,
            "exclude_from_scoring": excluded, "who_type": who_type, "author_type": author_type,
            "spend_signal": spend}


def test_incumbent_quality_pains_weigh_point_three():
    pure = compute_score(ClusterStats(20, 50, 5, 10, type_weight=0.3), 8, 3, W)
    # evidence (30 + 7.5 + 10) × 0.3 = 14.25; + reachability 16 − competition 6
    assert (pure.type_weight, pure.evidence, pure.total) == (0.3, 14.2, 24.2)
    rows = [member(i, pain_type="incumbent_quality") for i in range(3)] + [member(3)]
    assert compute_stats(rows, W).type_weight == pytest.approx((3 * 0.3 + 1) / 4)
    assert "incumbent_quality 0.3" in formula_markdown(W)


def test_stats_skip_non_colombian_pains_and_fall_back_to_author_type():
    rows = [member(0, who_type="business", spend="pago 300 mil"),
            member(1, author="a0", author_type="business"),  # same author, business via source
            member(2, excluded=1, who_type="business", spend="20 €"),
            member(3, who_type="consumer", author_type="business")]  # extraction wins
    stats = compute_stats(rows, W)
    assert (stats.size, stats.distinct_authors, stats.spend_count, stats.business_count,
            stats.excluded) == (3, 2, 1, 2, 1)


def test_known_inputs_for_a_reviewed_cluster():
    stats = ClusterStats(size=20, distinct_authors=50, spend_count=5, business_count=10)
    b = compute_score(stats, reachability=8, competition=3, w=W, reviewed=True)
    # frequency saturates at 50 authors → 30; spend 30×5/20; b2b 20×10/20; 20×0.8; −20×0.3
    assert (b.frequency, b.spend, b.b2b, b.reachability, b.competition) == (30, 7.5, 10, 16, 6)
    assert b.total == 57.5 and not b.capped and b.reviewed


def test_frequency_is_log_scaled_on_distinct_authors():
    def frequency(authors: int) -> float:
        return compute_score(ClusterStats(100, authors, 0, 0), 10, 0, W).frequency

    assert frequency(0) == 0
    assert frequency(1) == round(30 * math.log(2) / math.log(51), 1) == 5.3
    assert frequency(5) == round(30 * math.log(6) / math.log(51), 1) == 13.7
    assert frequency(50) == frequency(5000) == 30


def test_unreviewed_cluster_is_capped_at_20():
    b = compute_score(ClusterStats(30, 60, 30, 30), reachability=0, competition=0, w=W)
    assert (b.uncapped, b.total, b.capped) == (80, 20, True)


def test_competition_is_subtracted_and_score_floors_at_zero():
    b = compute_score(ClusterStats(10, 1, 0, 0), reachability=1, competition=10, w=W)
    assert b.competition == 20 and b.total == 0


def test_manual_inputs_are_clamped_to_0_10():
    b = compute_score(ClusterStats(10, 1000, 10, 10), reachability=15, competition=-3, w=W)
    assert (b.reachability, b.competition, b.total) == (20, 0, 100)


def test_weights_come_from_config_and_are_printed():
    w = ScoreWeights(spend=50, unreachable_cap=10)
    b = compute_score(ClusterStats(10, 1, 10, 0), 0, 0, w)
    assert (b.spend, b.total) == (50, 10)
    assert "50 × pains_with_spend_signal" in formula_markdown(w)
    assert "min(score, 10)" in formula_markdown(w)


def test_resolve_merges_follows_chains_and_reports_stale_ids():
    owner, warnings = resolve_merges({1, 2, 3, 4}, {1: [2], 3: [4], 2: [3, 99], 98: [1]})
    assert owner == {1: 1, 2: 1, 3: 1, 4: 1}
    assert len(warnings) == 2  # 99 and 98 aren't in the latest run


def test_score_latest_applies_merges_and_reviews(conn, tmp_path):
    a = [seed_pain(conn, 1, author="x", who_type="business", spend="pago 300 mil"),
         seed_pain(conn, 2, author="x"),  # same author: one distinct author
         seed_pain(conn, 3, author="y")]
    b = [seed_pain(conn, 4, author="z", who_type="business"), seed_pain(conn, 5, author="w")]
    noise = [seed_pain(conn, 6, author="v")]
    ca, cb = seed_cluster(conn, a), seed_cluster(conn, b)
    cn = seed_cluster(conn, noise, noise=True)
    (tmp_path / "merges.yaml").write_text(f"merges:\n  {ca}: [{cb}]\n")
    (tmp_path / "reachability.yaml").write_text(
        f"clusters:\n  {ca}: {{reachability: 5, competition: 2}}\n")

    scored = score_latest(conn, tmp_path, W)

    assert [s["id"] for s in scored] == [ca]
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM clusters")}
    merged = rows[ca]
    assert (merged["size"], merged["distinct_authors"], merged["spend_count"]) == (5, 4, 1)
    assert merged["b2b_ratio"] == 2 / 5
    expected = compute_score(ClusterStats(5, 4, 1, 2), 5, 2, W, reviewed=True)
    assert merged["score"] == expected.total
    assert rows[cb]["merged_into"] == ca and rows[cb]["score"] is None
    assert rows[cn]["score"] is None and rows[cn]["size"] == 1

    (tmp_path / "merges.yaml").write_text("merges: {}\n")  # undo: scoring is idempotent
    score_latest(conn, tmp_path, W)
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM clusters")}
    assert rows[cb]["merged_into"] is None and rows[cb]["score"] is not None
    assert rows[ca]["size"] == 3


def test_non_colombian_pains_stay_in_the_cluster_but_out_of_the_score(conn, tmp_path):
    mixed = seed_cluster(conn, [seed_pain(conn, 1, author="x"), seed_pain(conn, 2, author="y"),
                                seed_pain(conn, 3, author="z", country="mx")])
    abroad = seed_cluster(conn, [seed_pain(conn, 4, author="p", country="pe")])
    score_latest(conn, tmp_path, W)
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM clusters")}
    assert (rows[mixed]["size"], rows[mixed]["distinct_authors"]) == (2, 2)
    assert len(db.member_details(conn, [mixed])) == 3  # kept as a member
    assert rows[abroad]["score"] is None  # nothing left to score
