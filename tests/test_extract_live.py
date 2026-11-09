"""Live check against hand-labeled Colombian Spanish texts; used to pick the cheapest model.

    uv run pytest -m live                                   # llm.extract_model from settings
    NICHE_MINER_TEST_MODEL=claude-haiku-4-5 NICHE_MINER_TEST_EFFORT=none uv run pytest -m live

Costs a fraction of a cent per run."""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from niche_miner import db
from niche_miner.extract import Extractor
from niche_miner.llm import LLM
from niche_miner.models import RawSignal
from niche_miner.settings import load_settings

pytestmark = pytest.mark.live
ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests" / "fixtures" / "extraction_cases.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def extracted():
    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    settings = load_settings(ROOT / "config")
    model = os.environ.get("NICHE_MINER_TEST_MODEL") or settings.llm.extract_model
    if effort := os.environ.get("NICHE_MINER_TEST_EFFORT"):
        settings.llm.effort = None if effort == "none" else effort
    conn = db.connect(":memory:")
    db.insert_signals(conn, [
        RawSignal(niche_tag="fixtures", source=c["source"], external_id=c["id"],
                  url=c.get("url"), text=c["text"]) for c in CASES])
    stats = Extractor(conn, LLM(conn, settings.llm, "live"), settings.llm, model).run()
    cost = sum(u["cost_usd"] for u in db.llm_usage(conn, "live"))
    print(f"\n{model} (effort={settings.llm.effort}): {stats} — ${cost:.5f} "
          f"(≈ ${1000 * cost / len(CASES):.2f} per 1,000 signals)")
    rows = conn.execute("""SELECT rs.external_id, ep.* FROM extracted_pains ep
                           JOIN raw_signals rs ON rs.id = ep.raw_signal_id""").fetchall()
    return {r["external_id"]: r for r in rows}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_extraction_matches_labels(extracted, case):
    row = extracted[case["id"]]
    expect = case["expect"]
    assert row["error"] is None, row["error"]
    assert bool(row["is_complaint"]) == expect["is_complaint"], dict(row)
    if "who_type" in expect:
        assert row["who_type"] == expect["who_type"], dict(row)
    if "spend" in expect:
        assert (row["spend_signal"] is not None) == expect["spend"], dict(row)
    if "min_severity" in expect:
        assert row["severity"] >= expect["min_severity"], dict(row)
    for field in ("pain_type", "country"):
        if field in expect:
            assert row[field] == expect[field], dict(row)
    if expect["is_complaint"]:
        assert row["pain"] and row["pain_type"], dict(row)
    assert not (row["who"] or "").casefold().startswith("usuari"), dict(row)
    assert "." not in (row["current_tool"] or "").replace("manual/Excel/WhatsApp", ""), dict(row)
