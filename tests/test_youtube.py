"""YouTube parsers on synthetic API responses, and the collector over a fake API (no network)."""

import copy
import io
import json
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from niche_miner import db
from niche_miner.collectors import youtube
from niche_miner.collectors.base import SkipTarget, hash_author
from niche_miner.collectors.pii import scrub
from niche_miner.collectors.youtube import YouTubeCollector, parse_threads, video_id
from niche_miner.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures"
VIDEO = json.loads((FIXTURES / "youtube_video.json").read_text("utf-8"))
THREADS = json.loads((FIXTURES / "youtube_threads.json").read_text("utf-8"))
VID, OWNER, KEY = "AbCdEfGhI_1", "UCownerChannel000000000", "test-key-0123456789"
TITLE = "Nómina electrónica paso a paso & errores comunes"


@pytest.mark.parametrize("target", [
    VID, f" {VID} ", f"https://www.youtube.com/watch?v={VID}&t=42s", f"https://youtu.be/{VID}",
    f"https://www.youtube.com/shorts/{VID}"])
def test_video_id_from_ids_and_urls(target):
    assert video_id(target) == VID


def test_bad_video_id_is_skipped():
    with pytest.raises(SkipTarget):
        video_id("https://www.youtube.com/@canal")


def test_parse_threads_skips_the_owner_and_scrubs_people():
    signals = list(parse_threads(THREADS, VID, "n", OWNER))
    assert [s.external_id for s in signals] == [
        "UgxThreadOne", "UgxThreadOne.reply2", "UgxThreadTwo", "UgxThreadThree"]
    first, reply, _, anonymous = signals
    assert first.text.endswith("pago la licencia anual. Escríbanme a [correo]")
    assert reply.text == ("@usuario a mí me pasa igual desde enero, me toca hacer la nómina en "
                          "Excel y subirla a mano. Llámeme al [teléfono]")
    assert first.url == f"https://www.youtube.com/watch?v={VID}&lc=UgxThreadOne"
    assert first.author_hash == hash_author("youtube", "channel:UCmaria00000000000000000")
    assert anonymous.author_hash == hash_author("youtube", "comment:UgxThreadThree")
    assert "&amp;" not in anonymous.text and "como en el video & revisé" in anonymous.text
    assert first.posted_at == datetime(2026, 5, 4, 13, 20, tzinfo=UTC)
    assert all(s.author_type == "unknown" and s.source == "youtube" for s in signals)


def test_names_are_never_stored():
    dumped = "".join(s.model_dump_json() for s in parse_threads(THREADS, VID, "n", OWNER))
    for name in ("Maria", "Pedro", "AnaPyme", "maria.ejemplo", "310 555 0123"):
        assert name not in dumped


def test_replies_can_be_left_out():
    signals = parse_threads(THREADS, VID, "n", OWNER, include_replies=False)
    assert "UgxThreadOne.reply2" not in [s.external_id for s in signals]


@pytest.mark.parametrize(("text", "expected"), [
    ("@@Juan-Pérez_9 gracias", "@usuario gracias"),
    ("como dijo @ana.maria.", "como dijo @usuario."),
    ("escribe a rh@empresa.com.co", "escribe a [correo]"),
    ("cuesta 20 USD @ mes", "cuesta 20 USD @ mes"),
])
def test_scrub_mentions(text, expected):
    assert scrub(text) == expected


# --- collector over a fake API ----------------------------------------------------------------

def http_error(code: int, reason: str, message: str = "error") -> HTTPError:
    body = json.dumps({"error": {"code": code, "message": message,
                                 "errors": [{"reason": reason, "message": message}]}})
    return HTTPError("https://x", code, message, Message(), io.BytesIO(body.encode()))


def page_two() -> dict:
    page = copy.deepcopy(THREADS)
    page.pop("nextPageToken")
    page["items"] = page["items"][:1]
    comment = page["items"][0]["snippet"]["topLevelComment"]
    comment["id"] = "UgxThreadFour"
    comment["snippet"]["textDisplay"] = ("La factura electrónica se demora horas en validar y "
                                         "los clientes me reclaman todos los días.")
    page["items"][0].pop("replies")
    return page


@pytest.fixture
def api(monkeypatch):
    """(resource, video id, page token) -> JSON body, or an HTTPError to raise."""
    monkeypatch.setenv("YOUTUBE_API_KEY", KEY)
    responses: dict = {("videos", VID, None): VIDEO, ("commentThreads", VID, None): THREADS,
                       ("commentThreads", VID, "PAGE2"): page_two()}
    requests: list = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        parts = urlsplit(request.full_url)
        query = {k: v[0] for k, v in parse_qs(parts.query).items()}
        key = (parts.path.rsplit("/", 1)[-1], query.get("id") or query.get("videoId"),
               query.get("pageToken"))
        body = responses.get(key, {"items": []})
        if isinstance(body, Exception):
            raise body
        return io.BytesIO(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(youtube, "urlopen", fake_urlopen)
    return responses, requests


@pytest.fixture
def fast():
    settings = Settings()
    settings.collectors.backoff_s = 0
    settings.collectors.youtube.min_interval_s = 0
    return settings


CFG = {"tag": "contadores_pymes", "youtube_video_ids": [f"https://youtu.be/{VID}"]}


def test_collects_comments_with_the_title_added_after_filtering(api, fast):
    _, requests = api
    signals = YouTubeCollector(fast, known_ids={"UgxThreadThree"}).collect(CFG)
    # "Gracias!!" is too short on its own; the title must not rescue it.
    assert [s.external_id for s in signals] == ["UgxThreadOne", "UgxThreadOne.reply2",
                                                "UgxThreadFour"]
    assert all(s.text.startswith(f"Video: {TITLE}\n\n") for s in signals)
    assert [urlsplit(r.full_url).path for r in requests] == [
        "/youtube/v3/videos", "/youtube/v3/commentThreads", "/youtube/v3/commentThreads"]
    assert all(r.get_header("X-goog-api-key") == KEY for r in requests)
    assert not any(KEY in r.full_url for r in requests)


def test_no_api_key_skips_the_source(api, fast, monkeypatch):
    _, requests = api
    monkeypatch.delenv("YOUTUBE_API_KEY")
    assert YouTubeCollector(fast).collect(CFG) == [] and requests == []


def test_page_cap(api, fast):
    fast.collectors.youtube.max_pages_per_video = 1
    assert "UgxThreadFour" not in [s.external_id for s in YouTubeCollector(fast).collect(CFG)]


def test_missing_video_and_disabled_comments_skip_only_that_video(api, fast):
    responses, _ = api
    other = "ZzZzZzZzZ_9"
    responses[("videos", other, None)] = VIDEO
    responses[("commentThreads", other, None)] = http_error(403, "commentsDisabled")
    cfg = {**CFG, "youtube_video_ids": ["NoSuchVid_0", other, VID]}
    assert len(YouTubeCollector(fast).collect(cfg)) == 4


def test_quota_exceeded_stops_the_source_and_keeps_what_was_fetched(api, fast):
    responses, requests = api
    responses[("commentThreads", VID, "PAGE2")] = http_error(403, "quotaExceeded")
    cfg = {**CFG, "youtube_video_ids": [VID, "ZzZzZzZzZ_9"]}
    assert len(YouTubeCollector(fast).collect(cfg)) == 3
    assert len(requests) == 3  # nothing after the quota error, and no retries


def test_invalid_key_stops_the_source_without_leaking_it(api, fast, caplog):
    responses, requests = api
    responses[("videos", VID, None)] = http_error(400, "badRequest", f"API key not valid {KEY}")
    assert YouTubeCollector(fast).collect(CFG) == [] and len(requests) == 1
    assert KEY not in caplog.text and "API key not valid" in caplog.text


def test_server_errors_are_retried(api, fast):
    responses, requests = api
    responses[("videos", VID, None)] = http_error(503, "backendError")
    YouTubeCollector(fast).collect(CFG)
    assert len(requests) == fast.collectors.retries + 1


def test_recollecting_adds_nothing(api, fast, conn):
    first = db.insert_signals(conn, YouTubeCollector(fast).collect(CFG))
    known = db.external_ids(conn, "youtube")
    second = db.insert_signals(conn, YouTubeCollector(fast, known_ids=known).collect(CFG))
    assert (first, second) == (4, 0)
