"""Google Play parser + collector on the saved JSON fixture (no network)."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from urllib.error import URLError

import pytest
from google_play_scraper.exceptions import NotFoundError

from niche_miner import db
from niche_miner.collectors import google_play
from niche_miner.collectors.base import hash_author
from niche_miner.collectors.google_play import GooglePlayCollector, parse_review
from niche_miner.settings import GooglePlaySettings, Settings

FIXTURE = Path(__file__).parent / "fixtures" / "google_play_reviews.json"
CFG = GooglePlaySettings()
# Fixture reviews that survive parsing + filters: 3 = short, 4 = English, 5 = 5 stars,
# 7 = spam, 8 = no text, 9 = duplicate of 1.
EXPECTED_KEPT = {f"00000000-0000-4000-8000-0000000000{n:02d}" for n in (1, 2, 6, 10)}
APP_CFG = {"tag": "contadores_pymes", "google_play_apps": ["com.test.app"]}


def load_reviews() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_parse_review_maps_fields():
    raw = load_reviews()[0]
    signal = parse_review(raw, "com.siigo.nubeapp", "contadores_pymes", CFG)
    assert signal.external_id == raw["reviewId"]
    assert signal.source == "google_play"
    assert signal.niche_tag == "contadores_pymes"
    assert signal.url == ("https://play.google.com/store/apps/details?id=com.siigo.nubeapp"
                          f"&hl=es&gl=CO&reviewId={raw['reviewId']}")
    assert signal.text == raw["content"]
    assert signal.author_type == "unknown"
    assert signal.posted_at is not None and signal.posted_at.utcoffset().total_seconds() == 0


def test_reviewer_name_is_hashed_and_never_stored():
    raw = load_reviews()[0]
    signal = parse_review(raw, "com.test.app", "n", CFG)
    assert signal.author_hash == hashlib.sha256(f"google_play:{raw['userName']}".encode()).hexdigest()
    dumped = signal.model_dump_json()
    assert raw["userName"] not in dumped
    assert raw["replyContent"] not in dumped  # developer replies repeat the reviewer's name


def test_anonymous_reviewer_hashes_the_review_id():
    raw = next(r for r in load_reviews() if r["userName"] == "Un usuario de Google")
    signal = parse_review(raw, "com.test.app", "n", CFG)
    assert signal.author_hash == hash_author("google_play", raw["reviewId"])


def test_high_scores_and_empty_reviews_are_skipped():
    reviews = load_reviews()
    five_star = next(r for r in reviews if r["score"] == 5)
    no_text = next(r for r in reviews if r["content"] is None)
    assert parse_review(five_star, "com.test.app", "n", CFG) is None
    assert parse_review(no_text, "com.test.app", "n", CFG) is None


def fake_store(monkeypatch, apps: dict[str, list[dict]], failures: int = 0) -> dict:
    """Serve fixture reviews through the library's app()/reviews() interface."""
    state = {"failures": failures, "review_calls": 0}

    def fake_app(app_id, lang, country):
        if app_id not in apps:
            raise NotFoundError("App not found(404).")
        return {"title": app_id, "ratings": len(apps[app_id])}

    def fake_reviews(app_id, lang, country, sort, count, filter_score_with,
                     continuation_token=None):
        state["review_calls"] += 1
        if state["failures"]:
            state["failures"] -= 1
            raise URLError("temporary failure")
        matching = [r for r in apps[app_id] if r["score"] == filter_score_with]
        start = continuation_token or 0
        page = matching[start:start + count]
        return [{**r, "at": datetime.fromisoformat(r["at"])} for r in page], start + len(page)

    monkeypatch.setattr(google_play, "app", fake_app)
    monkeypatch.setattr(google_play, "reviews", fake_reviews)
    return state


@pytest.fixture
def fast():
    settings = Settings()
    settings.collectors.backoff_s = 0
    settings.collectors.google_play.min_interval_s = 0
    return settings


def test_collector_applies_shared_filters(monkeypatch, fast):
    fake_store(monkeypatch, {"com.test.app": load_reviews()})
    signals = GooglePlayCollector(fast).collect(APP_CFG)
    assert {s.external_id for s in signals} == EXPECTED_KEPT
    assert all(s.lang == "es" and len(s.text) >= 40 for s in signals)


def test_limit_caps_signals_per_source(monkeypatch, fast):
    fake_store(monkeypatch, {"com.test.app": load_reviews()})
    assert len(GooglePlayCollector(fast, limit=2).collect(APP_CFG)) == 2


def test_unused_limit_rolls_over_to_later_targets(monkeypatch, fast):
    template = load_reviews()[0]
    many = [{**template, "reviewId": f"big-{i}",
             "content": f"La aplicación falla al generar el reporte número {i} y nadie responde."}
            for i in range(10)]
    fake_store(monkeypatch, {"small": [template], "big": many})
    signals = GooglePlayCollector(fast, limit=6).collect(
        {"tag": "n", "google_play_apps": ["small", "big"]})
    assert len(signals) == 6  # small yields 1 of its 3, big gets the remaining 5


def test_unknown_app_is_skipped_and_others_still_collected(monkeypatch, fast):
    fake_store(monkeypatch, {"com.test.app": load_reviews()})
    cfg = {"tag": "n", "google_play_apps": ["com.wrong.id", "com.test.app"]}
    assert {s.external_id for s in GooglePlayCollector(fast).collect(cfg)} == EXPECTED_KEPT


def test_transient_errors_are_retried(monkeypatch, fast):
    state = fake_store(monkeypatch, {"com.test.app": load_reviews()}, failures=2)
    assert {s.external_id for s in GooglePlayCollector(fast).collect(APP_CFG)} == EXPECTED_KEPT
    assert state["review_calls"] == 4  # 2 failures + one page per score


def test_errors_after_retries_skip_the_target(monkeypatch, fast):
    fake_store(monkeypatch, {"com.test.app": load_reviews()}, failures=99)
    assert GooglePlayCollector(fast).collect(APP_CFG) == []


def test_missing_config_key_skips_the_source(fast):
    assert GooglePlayCollector(fast).collect({"tag": "n"}) == []


def test_recollecting_creates_no_duplicates(monkeypatch, fast, conn):
    fake_store(monkeypatch, {"com.test.app": load_reviews()})
    first = db.insert_signals(conn, GooglePlayCollector(fast).collect(APP_CFG))
    second = db.insert_signals(conn, GooglePlayCollector(fast).collect(APP_CFG))
    assert (first, second) == (len(EXPECTED_KEPT), 0)
    assert conn.execute("SELECT COUNT(*) FROM raw_signals").fetchone()[0] == len(EXPECTED_KEPT)
