import json
from datetime import date

from helpers import seed_cluster, seed_pain

from niche_miner import db
from niche_miner.cluster import NOISE_LABEL
from niche_miner.report import write_report
from niche_miner.score import score_latest


def _seed(conn):
    a = [seed_pain(conn, i, author=f"a{i}", who="Contador independiente", who_type="business",
                   spend="le pago 300 mil" if i == 0 else None,
                   text=f"Le pago 300 mil al mes a un contador porque no concilia ({i}).")
         for i in range(6)]
    b = [seed_pain(conn, 10 + i, author=f"b{i}", who="dueño de restaurante", tool=None,
                   pain_type="incumbent_quality") for i in range(5)]
    b.append(seed_pain(conn, 16, author="mx", country="mx"))
    first = seed_cluster(conn, a, label="Conciliación bancaria manual consume días")
    second = seed_cluster(conn, b, label="Pedidos por WhatsApp se pierden")
    seed_cluster(conn, [seed_pain(conn, 20, author="z")], label=NOISE_LABEL, noise=True)
    db.log_llm_call(conn, "run-1", "extract", "claude-haiku-5-5", 1000, 500, 0.012, 12)
    return first, second


def test_report_sections(conn, tmp_path, settings):
    first, second = _seed(conn)
    review = tmp_path / "review"
    review.mkdir()
    (review / f"{first}.json").write_text(json.dumps({
        "fetched_at": "2026-11-09T10:00:00+00:00",
        "competition": {"score": 6, "rationale": "Siigo concilia, pero a medias.",
                        "incumbents": [{"name": "Siigo", "country": "CO", "fit": "exact"}]}}))
    (review / "niche_test_niche.json").write_text(json.dumps({
        "score": 8, "rationale": "2 channel type(s) with signal → 5 +1 strong (job_board) +2 "
        "creator comments = 8", "fetched_at": "2026-11-09T10:00:00+00:00",
        "types": {"job_board": {"signal": True, "strong": True, "posts": 918,
                                "queries": ["auxiliar contable"], "cities": ["bogota-dc"]},
                  "gremio": {"signal": True, "query": "test niche asociación Colombia",
                             "gremios": [{"name": "INCP", "member_directory": False}]}},
        "creator_comments": {"points": 2, "channels": [{"name": "YouTube X", "audience": 35200}]}}))
    (tmp_path / "niches.yaml").write_text("niches:\n  - tag: test_niche\n    buyer_roles: [dueño]\n")
    (tmp_path / "reachability.yaml").write_text(f"clusters:\n  {first}: {{competition: 4}}\n")
    score_latest(conn, tmp_path, settings.scoring, review)

    path = write_report(conn, settings, tmp_path / "reports", today=date(2026, 11, 8),
                        review_dir=review)
    md = path.read_text(encoding="utf-8")

    assert path.name == "2026-11-08.md"
    assert "## Scoring formula" in md
    assert "Σ_document min(ln(1 + authors_document), ln(1 + 30)) / 15" in md
    assert "| id | label | status | pains | docs | sources | top doc | authors | freq. sum |" in md
    assert "Σ min(ln(1 + authors per document), ln(1 + 30)) = " in md
    assert "below the ceiling 15; 0 document(s) at the 30-author cap" in md
    assert md.index("Conciliación bancaria") < md.index("Pedidos por WhatsApp")  # ranked
    assert "| # | id | theme | score | old freq. | co-only |" in md  # top table
    assert md.count("**Hypothesis:**") == 2
    assert "](https://example.com/review/" in md  # quotes link to their source
    assert "contador independiente (6)" in md and "(sin dato) (5)" in md
    assert "le pago 300 mil" in md  # spend signals shown
    assert "| Pain-type weight | incumbent_quality 5 | ×0.30 |" in md
    assert "1 more from outside Colombia, not scored" in md
    assert "Complaints from outside Colombia (kept, not scored): 1 (mx 1)" in md
    assert "per 1,000 signals" in md and "noise, not scored" in md
    # niche reachability by type, then niche proxy × who_match per cluster
    assert "**`test_niche` — reachability_proxy 8/10**" in md
    assert "- job_board: 918 Computrabajo posts in 30 days (auxiliar contable; bogota-dc) — strong" in md
    assert "- gremio: 1 found for “test niche asociación Colombia”: INCP" in md
    assert "creator_comments (+2, never a type): YouTube X (35.2k)" in md
    assert "0/10 (automated 0 = niche 8 × who_match 0.00, no override)" in md  # contadores
    assert "8/10 (automated 8 = niche 8 × who_match 1.00, no override)" in md  # dueños
    assert "**who_match:** 0.00 — 0 pains from buyer roles, 0 without `who` (×0.7), 6 from" in md
    assert "4/10 (automated 6, override 4)" in md
    assert "**Incumbents** (automated 2026-11-09, competition 6/10): Siigo (CO, exact)" in md
    assert "Automated review: 1 of 2 active clusters" in md
    # flagged but still ranked; the second has no competition review and isn't reviewed
    assert "🔴 no visible audience" in md and "_No automated review yet" in md
    assert f"- `{second}` Pedidos por WhatsApp se pierden (automated: reachability 8)" in md
    assert f"  {second}: {{reachability: null, competition: null, status: active" in md
    assert f"- `{first}`" not in md.split("## Not yet reviewed")[1]  # override = reviewed
    assert f"add `  {first}: {{status: not_a_niche}}`" in md  # one-line status marker
    assert "⚠ one document" not in md  # every seeded pain has its own review URL


def test_not_a_niche_leaves_the_top_list_but_stays_in_the_table(conn, tmp_path, settings):
    first, second = _seed(conn)
    (tmp_path / "reachability.yaml").write_text(
        f"clusters:\n  {first}: {{status: not_a_niche}}\n")
    score_latest(conn, tmp_path, settings.scoring)
    md = write_report(conn, settings, tmp_path, today=date(2026, 1, 1)).read_text(encoding="utf-8")
    top = md.split("## All clusters")[0]
    assert "### 1. Pedidos por WhatsApp se pierden" in top
    assert "### " + "1. Conciliación" not in top and "marked `not_a_niche`" in top
    assert f"| {first} | Conciliación bancaria manual consume días | not_a_niche |" in md


def test_one_document_concentration_is_flagged(conn, tmp_path, settings):
    pains = [seed_pain(conn, i, author=f"a{i}") for i in range(5)]
    conn.execute("UPDATE raw_signals SET source = 'youtube', "
                 "url = 'https://www.youtube.com/watch?v=V1&lc=c' || id")
    cid = seed_cluster(conn, pains, label="Habilitación DIAN confusa")
    score_latest(conn, tmp_path, settings.scoring)
    md = write_report(conn, settings, tmp_path, today=date(2026, 1, 1)).read_text(encoding="utf-8")
    assert "⚠ one document 100%" in md
    assert "**One document holds 5 of 5 pains (100%):** https://www.youtube.com/watch?v=V1" in md
    assert f"| {cid} | Habilitación DIAN confusa | active | 5 | 1 | 1 | 100% |" in md


def test_report_without_clusters_is_still_written(conn, tmp_path, settings):
    path = write_report(conn, settings, tmp_path, today=date(2026, 1, 1))
    assert "No scored clusters yet" in path.read_text(encoding="utf-8")
