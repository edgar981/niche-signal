"""Automated review offline: the niche reachability ladder, who_match, parsers, and one full
run with fakes for Apify, the Play Store, Computrabajo, YouTube and Claude."""

import json
from datetime import UTC, datetime

import pytest
from helpers import FakeLLM, seed_cluster, seed_pain

from niche_miner import autoreview, competition
from niche_miner.audience import (creator_points, matches_role, niche_score, parse_offer_count,
                                  tiktok_accounts, who_match)
from niche_miner.competition import build_queries, parse_search
from niche_miner.score import score_latest


def types(*with_signal, strong=()):
    return {t: {"signal": t in with_signal, "strong": t in strong}
            for t in ("job_board", "fb_groups", "gremio", "whatsapp_community")}


@pytest.mark.parametrize(("signal", "strong", "creator", "score"), [
    ((), (), 0, 0),
    ((), (), 2, 2),  # creator comments alone are never a type
    (("job_board",), (), 0, 3),
    (("job_board", "gremio"), (), 0, 5),
    (("job_board", "gremio", "fb_groups"), (), 0, 7),
    (("job_board", "gremio"), ("job_board",), 1, 7),
    (("job_board", "gremio", "fb_groups"), ("job_board", "gremio", "fb_groups"), 2, 10),
    (("job_board", "gremio", "fb_groups", "whatsapp_community"), ("job_board",), 2, 10),  # cap
    (("job_board",), ("gremio",), 0, 3),  # strong without signal doesn't count
])
def test_niche_ladder(signal, strong, creator, score):
    assert niche_score(types(*signal, strong=strong), {"points": creator})[0] == score


def test_creator_comments_add_at_most_two():
    assert creator_points([], [], 20_000) == 0
    assert creator_points([], ["cuentate.co"], 20_000) == 1  # TikTok, no follower counts
    assert creator_points([5_000, None], [], 20_000) == 1
    assert creator_points([35_000, 205_000], ["x"], 20_000) == 2


def test_who_match_discounts_non_buyers_and_weighs_unknown():
    roles = ["dueño", "empresa", "persona natural"]
    assert matches_role("Dueña de ferretería", ["dueña"]) and matches_role("Empresario", roles)
    assert matches_role("persona natural revendedora", roles)
    assert not matches_role("contador independiente", roles)
    rows = [{"who": w} for w in ("dueño de negocio", "contador", None, "", "empresa textil")]
    share, counts = who_match(rows, roles, 0.7)
    assert counts == {"buyer": 2, "unknown": 2, "other": 1}
    assert share == pytest.approx((2 + 0.7 * 2) / 5)
    assert who_match(rows, [], 0.7)[0] == 1.0  # no buyer_roles configured: no discount


def test_parsers():
    html = "<h1 class='t'>\n <span>585</span> Ofertas de trabajo de auxiliar contable en Bogotá</h1>"
    assert parse_offer_count(html) == 585
    assert parse_offer_count("<h1>Ofertas de trabajo de nada</h1>") == 0
    items = [{"searchQuery": {"term": "conciliación Colombia"},
              "organicResults": [{"title": "Siigo", "url": "https://siigo.com", "description": "d"},
                                 {"title": "no url"}]},
             {"searchQuery": {}, "organicResults": [{"url": "https://x"}]}]
    assert parse_search(items, 10) == {"conciliación Colombia": [
        {"title": "Siigo", "url": "https://siigo.com", "snippet": "d"}]}
    assert build_queries("Conciliación", ["Colombia", "app"]) == ["Conciliación Colombia",
                                                                  "Conciliación app"]
    rows = [{"source": "apify_tiktok", "url": "https://www.tiktok.com/@contadora/video/1"},
            {"source": "youtube", "url": None}]
    assert tiktok_accounts(rows) == ["contadora"]


class FakeRun:
    usage_total_usd = 0.0135
    status, status_message = "SUCCEEDED", None
    id = "run-1"


def llm_reply(call):
    if "trade associations" in call["system"]:
        return json.dumps({"gremios": [{"name": "INCP", "url": "https://incp.org.co",
                                        "member_directory": False, "evidence": "result 1"}]})
    return json.dumps({"incumbents": [{"name": "Siigo", "country": "CO", "fit": "exact",
                                       "evidence": "siigo.com"}],
                       "competition": 8, "rationale": "Siigo does it."})


def test_review_writes_both_caches_and_score_uses_them(conn, tmp_path, settings, monkeypatch):
    whos = ["dueño de negocio", "dueño de negocio", "contador", None]
    cid = seed_cluster(conn, [seed_pain(conn, i, author=f"a{i}", who=w, niche="test_niche")
                              for i, w in enumerate(whos)], label="Conciliación bancaria manual")
    dropped = seed_cluster(conn, [seed_pain(conn, 10, author="b")], label="Otra cosa")
    (tmp_path / "reachability.yaml").write_text(
        f"clusters:\n  {dropped}: {{status: not_a_niche}}\n")
    (tmp_path / "niches.yaml").write_text(
        "niches:\n  - tag: test_niche\n    computrabajo_queries: [auxiliar contable]\n"
        "    buyer_roles: [dueño, empresa]\n")
    niches = {"test_niche": {"tag": "test_niche", "computrabajo_queries": ["auxiliar contable"],
                             "buyer_roles": ["dueño", "empresa"]}}
    started = []

    def fake_start(token, spec, run_input, max_items, cap, timeout):
        started.append(run_input)
        return [{"searchQuery": {"term": q}, "organicResults": [
            {"title": "Siigo", "url": "https://siigo.com", "description": "concilia"}]}
            for q in run_input["queries"].split("\n")], FakeRun()

    monkeypatch.setenv("APIFY_TOKEN", "test-token")
    monkeypatch.setattr(autoreview, "start_and_wait", fake_start)
    monkeypatch.setattr(autoreview, "play_search", lambda term, hits: [{"title": "Alegra"}])
    settings.review.search.actor = "apify/google-search-scraper"
    settings.review.search.input = {"queries": "{queries}", "countryCode": "co"}

    def reviewer(llm, day):
        r = autoreview.Reviewer(conn, settings, tmp_path, tmp_path / "review", llm, niches,
                                now=datetime(2026, 11, day, tzinfo=UTC))
        r.yt.api_key = ""
        monkeypatch.setattr(r, "_offer_count", lambda query, city: 30)  # 3 cities: 90 posts
        return r

    stats = reviewer(FakeLLM(llm_reply, llm_reply), 9).run()

    assert (stats.reviewed, stats.niches_reviewed, stats.search_usd) == (1, 1, 0.0135)
    assert set(started[0]["queries"].split("\n")) == {
        "Conciliación bancaria manual Colombia", "Conciliación bancaria manual app",
        "Conciliación bancaria manual software", "test niche asociación Colombia"}
    comp = json.loads((tmp_path / "review" / f"{cid}.json").read_text())
    assert comp["competition"]["score"] == 8 and "reachability" not in comp
    niche = json.loads((tmp_path / "review" / "niche_test_niche.json").read_text())
    assert niche["types"]["job_board"] | {"detail": None} == {
        "signal": True, "strong": True, "posts": 90, "queries": ["auxiliar contable"],
        "cities": ["bogota-dc", "medellin", "cali"], "detail": None}
    assert niche["types"]["gremio"]["gremios"][0]["name"] == "INCP"
    assert niche["score"] == 6  # two types (5) + strong job board (1)
    assert not (tmp_path / "review" / f"{dropped}.json").exists()  # not_a_niche is skipped

    again = reviewer(FakeLLM(), 20).run()  # inside cache_days: no calls at all
    assert (again.reviewed, again.cached, again.niches_reviewed, again.niches_cached) == (0, 1, 0, 1)

    score_latest(conn, tmp_path, settings.scoring, tmp_path / "review")
    row = conn.execute("SELECT reachability, competition, score_breakdown FROM clusters "
                       "WHERE id = ?", (cid,)).fetchone()
    b = json.loads(row["score_breakdown"])
    assert b["who_counts"] == {"buyer": 2, "unknown": 1, "other": 1}
    assert b["who_match"] == pytest.approx((2 + 0.7) / 4, abs=1e-3)
    assert (row["reachability"], row["competition"]) == (round(6 * 2.7 / 4, 1), 8)  # 4.0 and 8
    assert (b["niche_proxy"], b["reachability_from"]) == (6, "automated")


def test_competition_response_is_validated():
    llm = FakeLLM(json.dumps({"incumbents": [{"name": ""}], "competition": 14,
                              "rationale": "r"}), "not json")
    got = competition.assess(llm, "m", "label", ["dolor"], {}, [], [["Siigo", 3]])
    assert got == {"score": 10, "rationale": "r", "incumbents": []}
    assert "unavailable" in llm.calls[0]["user"]  # no Google results this run
    with pytest.raises(competition.LLMError):
        competition.assess(llm, "m", "label", [], {}, [], [])
