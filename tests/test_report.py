from datetime import date

from helpers import seed_cluster, seed_pain

from niche_miner import db
from niche_miner.cluster import NOISE_LABEL
from niche_miner.report import write_report
from niche_miner.score import score_latest


def test_report_sections(conn, tmp_path, settings):
    a = [seed_pain(conn, i, author=f"a{i}", who="Contador independiente", who_type="business",
                   spend="le pago 300 mil" if i == 0 else None,
                   text=f"Le pago 300 mil al mes a un contador porque no concilia ({i}).")
         for i in range(6)]
    b = [seed_pain(conn, 10 + i, author=f"b{i}", who="dueño de restaurante", tool=None,
                   pain_type="incumbent_quality") for i in range(5)]
    b.append(seed_pain(conn, 16, author="mx", country="mx"))
    first = seed_cluster(conn, a, label="Conciliación bancaria manual consume días")
    seed_cluster(conn, b, label="Pedidos por WhatsApp se pierden")
    seed_cluster(conn, [seed_pain(conn, 20, author="z")], label=NOISE_LABEL, noise=True)
    db.log_llm_call(conn, "run-1", "extract", "claude-haiku-5-5", 1000, 500, 0.012, 12)
    score_latest(conn, tmp_path, settings.scoring)

    path = write_report(conn, settings, tmp_path / "reports", today=date(2026, 11, 8))
    md = path.read_text(encoding="utf-8")

    assert path.name == "2026-11-08.md"
    assert "## Scoring formula" in md and "ln(1 + distinct_authors)" in md
    assert md.index("Conciliación bancaria") < md.index("Pedidos por WhatsApp")  # ranked
    assert md.count("**Hypothesis:**") == 2
    assert "](https://example.com/review/" in md  # quotes link to their source
    assert "contador independiente (6)" in md  # who distribution (normalized)
    assert "(sin dato) (5)" in md  # current_tool missing for cluster b
    assert "le pago 300 mil" in md  # spend signals shown
    assert "capped (reachability = 0" in md
    assert "| Pain-type weight | incumbent_quality 5 | ×0.30 |" in md
    assert "1 more from outside Colombia, not scored" in md
    assert "Complaints from outside Colombia (kept, not scored): 1 (mx 1)" in md
    assert "Pain types: missing_capability 8 · incumbent_quality 5" in md
    assert "per 1,000 signals" in md  # cost line
    assert "noise, not scored" in md  # noise row in the all-clusters table
    assert f"  {first}: {{reachability: 0, competition: 0}}" in md  # review stub


def test_report_without_clusters_is_still_written(conn, tmp_path, settings):
    path = write_report(conn, settings, tmp_path, today=date(2026, 1, 1))
    assert "No scored clusters yet" in path.read_text(encoding="utf-8")
