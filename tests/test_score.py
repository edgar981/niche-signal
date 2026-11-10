"""Scoring with known inputs: per-document frequency, b2b from known roles, overrides over
automated values, statuses, co-only scores, and merges applied from YAML."""

import json
import math

import pytest
from helpers import seed_cluster, seed_pain

from niche_miner import db
from niche_miner.reviews import load_overrides, resolve_merges
from niche_miner.score import ClusterStats, compute_score, compute_stats, formula_markdown, score_latest
from niche_miner.settings import ScoreWeights
from niche_miner.source_rules import source_key

W = ScoreWeights()
FULL = W.frequency_ceiling  # url_spread that earns full frequency points


def member(i: int, *, author: str | None = None, pain_type: str = "missing_capability",
           excluded: int = 0, who_type: str = "unknown", author_type: str = "unknown",
           spend: str | None = None, source: str = "youtube", doc: str = "v1") -> dict:
    url = f"https://www.youtube.com/watch?v={doc}&lc=c{i}" if source == "youtube" else doc
    return {"signal_id": i, "author_hash": author or f"a{i}", "pain_type": pain_type,
            "exclude_from_scoring": excluded, "who_type": who_type, "author_type": author_type,
            "spend_signal": spend, "source": source, "url": url}


def test_incumbent_quality_pains_weigh_point_three():
    pure = compute_score(ClusterStats(20, 50, 5, 10, type_weight=0.3, known_role=20,
                                      url_spread=FULL), 8, 3, W)
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
            stats.known_role, stats.excluded) == (3, 2, 1, 2, 3, 1)


def test_comments_and_reviews_are_grouped_by_their_document():
    assert source_key("youtube", "https://www.youtube.com/watch?v=abc&lc=XYZ", 1) == \
        "https://www.youtube.com/watch?v=abc"
    play = "https://play.google.com/store/apps/details?id=com.x&hl=es&gl=CO&reviewId=r1"
    assert source_key("google_play", play, 2) == "https://play.google.com/store/apps/details?id=com.x"
    assert source_key("computrabajo", None, 7) == "computrabajo:signal:7"


def test_frequency_sums_capped_documents_against_the_ceiling():
    def freq(rows: list[dict]):
        return compute_score(compute_stats(rows, W), 10, 0, W)

    one_video = [member(i, doc="v1") for i in range(11)]  # 11 authors under one video
    spread = [member(i, doc=f"v{i}") for i in range(6)]  # 6 videos, one author each
    b = freq(one_video)  # ln 12 / 15; old: ln 12 / ln 51
    assert (b.frequency, b.frequency_old, b.freq_sum, b.cap_hits) == (5.0, 19.0, 2.48, 0)
    assert freq(spread).frequency == round(30 * 6 * math.log(2) / 15, 1) == 8.3
    crowded = freq([member(i, doc="v1") for i in range(40)])  # 40 authors count as 30
    assert (crowded.freq_sum, crowded.cap_hits, crowded.frequency) == (3.43, 1, 6.9)
    wide = freq([member(i, doc=f"v{i}") for i in range(22)])  # 22 · ln 2 = 15.2 ≥ 15
    assert (wide.frequency, wide.above_ceiling) == (30, True)
    assert not crowded.above_ceiling
    stats = compute_stats(one_video + [member(20, doc="v2")], W)
    assert (stats.distinct_urls, stats.distinct_sources, stats.top_url_count) == (2, 1, 11)
    assert stats.top_url_share == pytest.approx(11 / 12)


def test_b2b_uses_known_roles_or_the_prior_when_too_few_are_known():
    sparse = ClusterStats(10, 10, 0, business_count=2, known_role=2, url_spread=FULL)
    b = compute_score(sparse, 0, 0, W, b2b_prior=0.6)
    assert (b.b2b_insufficient, b.b2b_share, b.b2b) == (True, 0.6, 12)
    known = ClusterStats(10, 10, 0, business_count=2, known_role=4, url_spread=FULL)
    b = compute_score(known, 0, 0, W, b2b_prior=0.6)
    assert (b.b2b_insufficient, b.b2b_share, b.b2b, b.b2b_known) == (False, 0.5, 10, 0.4)


def test_b2b_weight_zero_turns_the_component_off():
    off = W.model_copy(update={"b2b": 0})
    sparse = ClusterStats(10, 10, 0, business_count=2, known_role=0, url_spread=FULL)
    b = compute_score(sparse, 0, 0, off, b2b_prior=0.6)
    assert (b.b2b_insufficient, b.b2b_share, b.b2b) == (False, 0.0, 0)
    assert "b2b          = 0 (off" in formula_markdown(off)


def test_known_inputs_and_no_cap_when_reachability_is_zero():
    b = compute_score(ClusterStats(20, 50, 5, 10, known_role=20, url_spread=FULL), 8, 3, W)
    assert (b.frequency, b.spend, b.b2b, b.reachability, b.competition) == (30, 7.5, 10, 16, 6)
    assert b.total == 57.5
    unreachable = compute_score(ClusterStats(30, 60, 30, 30, known_role=30, url_spread=FULL),
                                0, 0, W)
    assert unreachable.total == 80  # ranked on evidence; flagged elsewhere, not capped


def test_competition_is_subtracted_and_inputs_are_clamped():
    b = compute_score(ClusterStats(10, 1, 0, 0, url_spread=math.log(2)), 1, 10, W)
    assert b.competition == 20 and b.total == 0
    b = compute_score(ClusterStats(10, 1000, 10, 10, known_role=10, url_spread=99), 15, -3, W)
    assert (b.reachability, b.competition, b.total) == (20, 0, 100)


def test_weights_come_from_config_and_are_printed():
    w = ScoreWeights(spend=50, b2b_min_known_share=0.5, no_audience_below=4)
    b = compute_score(ClusterStats(10, 1, 10, 0), 0, 0, w)
    assert (b.spend, b.total) == (50, 50)
    md = formula_markdown(w)
    assert "50 × pains_with_spend_signal" in md and "under 50% of pains" in md
    assert "reachability < 4" in md
    assert "Σ_document min(ln(1 + authors_document), ln(1 + 30)) / 15" in md
    assert "not run yet" in md
    assert "niche reachability_proxy × who_match" in md


def test_resolve_merges_follows_chains_and_reports_stale_ids():
    owner, warnings = resolve_merges({1, 2, 3, 4}, {1: [2], 3: [4], 2: [3, 99], 98: [1]})
    assert owner == {1: 1, 2: 1, 3: 1, 4: 1}
    assert len(warnings) == 2  # 99 and 98 aren't in the latest run


def test_overrides_parse_statuses_and_report_bad_values(tmp_path):
    path = tmp_path / "reachability.yaml"
    path.write_text("clusters:\n  1: {reachability: 12, competition: null, status: active}\n"
                    "  2: {status: not_a_niche}\n  3: {status: 'merged_into:1'}\n"
                    "  4: {status: archived, competition: high}\n  5: {note: checked}\n"
                    "  99: {reachability: 3}\n")
    overrides, merges, warnings = load_overrides(path, {1, 2, 3, 4, 5})
    assert overrides[1].reachability == 10 and overrides[1].competition is None
    assert overrides[2].status == "not_a_niche" and overrides[2].reviewed
    assert merges == {1: [3]} and overrides[3].merged_into == 1
    assert overrides[4].status == "active" and not overrides[4].reviewed
    assert overrides[5].reviewed  # a note alone marks it reviewed
    assert len(warnings) == 4  # clamped 12, bad status, bad number, stale 99


def _row(conn, cid):
    return conn.execute("SELECT * FROM clusters WHERE id = ?", (cid,)).fetchone()


def test_score_latest_applies_merges_and_overrides(conn, tmp_path):
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
    merged = _row(conn, ca)
    assert (merged["size"], merged["distinct_authors"], merged["spend_count"]) == (5, 4, 1)
    assert merged["b2b_ratio"] == 1.0  # 2 businesses among 2 known roles
    stats = compute_stats(db.member_details(conn, [ca, cb]), W)
    assert merged["score"] == compute_score(stats, 5, 2, W, b2b_prior=1.0).total
    breakdown = json.loads(merged["score_breakdown"])
    assert breakdown["reachability_from"] == "override" and breakdown["reviewed"]
    assert _row(conn, cb)["merged_into"] == ca and _row(conn, cb)["score"] is None
    assert _row(conn, cn)["score"] is None and _row(conn, cn)["size"] == 1

    (tmp_path / "merges.yaml").write_text("merges: {}\n")  # undo: scoring is idempotent
    (tmp_path / "reachability.yaml").write_text(f"clusters:\n  {cb}: {{status: 'merged_into:{ca}'}}\n")
    score_latest(conn, tmp_path, W)
    assert _row(conn, cb)["merged_into"] == ca  # the same merge, from a status


def test_automated_values_fill_in_unless_overridden(conn, tmp_path):
    whos = ("dueño de negocio", "contador", "dueño de negocio")
    ids = [seed_cluster(conn, [seed_pain(conn, 10 * k + i, author=f"a{k}{i}", who=who)
                               for i in range(3)]) for k, who in enumerate(whos)]
    review = tmp_path / "review"
    review.mkdir()
    (review / "niche_test_niche.json").write_text(json.dumps({"score": 7, "types": {}}))
    for cid in ids:
        (review / f"{cid}.json").write_text(json.dumps({
            "fetched_at": "2026-11-09T00:00:00+00:00",
            "competition": {"score": 4, "rationale": "r", "incumbents": [{"name": "Siigo"}]}}))
    (tmp_path / "niches.yaml").write_text("niches:\n  - tag: test_niche\n    buyer_roles: [dueño]\n")
    (tmp_path / "reachability.yaml").write_text(
        f"clusters:\n  {ids[0]}: {{competition: 9}}\n  {ids[2]}: {{status: not_a_niche}}\n")

    scored = {s["id"]: s for s in score_latest(conn, tmp_path, W, review)}

    first, second, third = (json.loads(_row(conn, cid)["score_breakdown"]) for cid in ids)
    assert (first["reachability_value"], first["reachability_from"]) == (7, "automated")
    assert (first["niche_proxy"], first["who_match"]) == (7, 1.0)
    assert (first["competition_value"], first["competition_from"]) == (9, "override")
    assert first["auto"]["competition"]["incumbents"][0]["name"] == "Siigo"
    assert second["reachability_value"] == 0 and second["who_counts"]["other"] == 3
    assert not first["no_audience"] and second["no_audience"]  # 7 × 0 < 3, no override
    assert third["status"] == "not_a_niche" and scored[ids[2]]["score"] > 0  # still scored


def test_co_only_score_leaves_out_unknown_countries(conn, tmp_path):
    cid = seed_cluster(conn, [seed_pain(conn, 1, author="x"), seed_pain(conn, 2, author="y"),
                              seed_pain(conn, 3, author="z", country="unknown"),
                              seed_pain(conn, 4, author="p", country="mx")])
    only_unknown = seed_cluster(conn, [seed_pain(conn, 5, author="q", country="unknown")])
    score_latest(conn, tmp_path, W)
    b = json.loads(_row(conn, cid)["score_breakdown"])
    assert _row(conn, cid)["size"] == 3 and b["co_only"]["size"] == 2
    assert json.loads(_row(conn, only_unknown)["score_breakdown"])["co_only"] is None


def test_non_colombian_pains_stay_in_the_cluster_but_out_of_the_score(conn, tmp_path):
    mixed = seed_cluster(conn, [seed_pain(conn, 1, author="x"), seed_pain(conn, 2, author="y"),
                                seed_pain(conn, 3, author="z", country="mx")])
    abroad = seed_cluster(conn, [seed_pain(conn, 4, author="p", country="pe")])
    score_latest(conn, tmp_path, W)
    assert (_row(conn, mixed)["size"], _row(conn, mixed)["distinct_authors"]) == (2, 2)
    assert len(db.member_details(conn, [mixed])) == 3  # kept as a member
    assert _row(conn, abroad)["score"] is None  # nothing left to score
