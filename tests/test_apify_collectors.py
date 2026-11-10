"""Apify collectors over a fake ApifyClient (no network, no credits), using the shipped config."""

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from apify_client.errors import ApifyApiError

from niche_miner import db
from niche_miner.collectors import apify_common
from niche_miner.collectors.apify_facebook import ApifyFacebookCollector
from niche_miner.collectors.apify_gmaps import ApifyGmapsCollector
from niche_miner.collectors.apify_tiktok import ApifyTikTokCollector
from niche_miner.settings import Settings, load_settings

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = Path(__file__).parents[1] / "config"
TOKEN = "apify_api_test_token_0123456789"
GMAPS_ACTOR, VIDEOS_ACTOR = "compass/crawler-google-places", "clockworks/free-tiktok-scraper"
COMMENTS_ACTOR, FB_ACTOR = "clockworks/tiktok-comments-scraper", "apify/facebook-groups-scraper"
CFG = {"tag": "contadores_pymes", "gmaps_queries": ["software contable Bogotá"],
       "tiktok_hashtags": ["#contadorescolombia"],
       "facebook_groups": ["https://www.facebook.com/groups/contadoresejemplo"]}


def load(name: str):
    return json.loads((FIXTURES / f"apify_{name}.json").read_text("utf-8"))


class FakeApiError(ApifyApiError):
    """ApifyApiError without an HTTP response behind it."""

    def __new__(cls, *args, **kwargs):
        return Exception.__new__(cls)

    def __init__(self, status_code: int, kind: str = "", message: str = "error"):
        Exception.__init__(self, message)
        self.status_code, self.type, self.message = status_code, kind, message


class FakeApify:
    """Stands in for ApifyClient. outputs: actor id -> items, an exception for start() to
    raise, or a function of the run input returning either."""

    def __init__(self):
        self.outputs: dict = {}
        self.starts: list[tuple[str, dict]] = []
        self.status, self.message, self.usage = "SUCCEEDED", None, 0.01
        self.tokens: list[str] = []
        self._items: dict[str, list] = {}

    def __call__(self, token: str):
        self.tokens.append(token)
        return self

    def actor(self, actor_id: str):
        def start(**kwargs):
            self.starts.append((actor_id, kwargs))
            out = self.outputs.get(actor_id, [])
            out = out(kwargs["run_input"]) if callable(out) else out
            if isinstance(out, Exception):
                raise out
            run_id = f"run{len(self.starts)}"
            self._items[run_id] = out
            return SimpleNamespace(id=run_id, default_dataset_id=run_id, status="READY",
                                   status_message=None, usage_total_usd=None)
        return SimpleNamespace(start=start)

    def run(self, run_id: str):
        def state(usage):
            return SimpleNamespace(id=run_id, default_dataset_id=run_id, status=self.status,
                                   status_message=self.message, usage_total_usd=usage)
        # Like Apify: the cost is only settled a moment after the run finishes.
        return SimpleNamespace(wait_for_finish=lambda **kwargs: state(None),
                               get=lambda **kwargs: state(self.usage))

    def dataset(self, dataset_id: str):
        return SimpleNamespace(iterate_items=lambda **kwargs: iter(self._items[dataset_id]))


@pytest.fixture
def apify(monkeypatch):
    monkeypatch.setenv("APIFY_TOKEN", TOKEN)
    fake = FakeApify()
    fake.outputs = {GMAPS_ACTOR: load("gmaps_places"), VIDEOS_ACTOR: load("tiktok_videos"),
                    COMMENTS_ACTOR: load("tiktok_comments"), FB_ACTOR: load("facebook_posts")}
    monkeypatch.setattr(apify_common, "ApifyClient", fake)
    return fake


@pytest.fixture
def settings():
    return load_settings(CONFIG)  # the shipped actors and inputs


def test_without_token_every_apify_source_is_skipped(apify, settings, monkeypatch):
    monkeypatch.delenv("APIFY_TOKEN")
    for cls in (ApifyGmapsCollector, ApifyTikTokCollector, ApifyFacebookCollector):
        assert cls(settings).collect(CFG) == []
    assert apify.starts == []


def test_unconfigured_actor_is_skipped(apify):
    assert ApifyGmapsCollector(Settings()).collect(CFG) == [] and apify.starts == []


def test_gmaps_input_caps_and_place_context(apify, settings, caplog):
    caplog.set_level("INFO")
    signals = ApifyGmapsCollector(settings).collect(CFG)
    actor, kwargs = apify.starts[0]
    assert actor == GMAPS_ACTOR and apify.tokens == [TOKEN]
    assert kwargs["run_input"]["searchStringsArray"] == ["software contable Bogotá"]
    assert kwargs["run_input"]["scrapeReviewsPersonalData"] is False
    # The actor refuses caps under $0.50; its input (10 places x 20 reviews) bounds the spend.
    assert kwargs["max_total_charge_usd"] == Decimal("0.5") and kwargs["max_items"] is None
    # "Malo" is too short on its own; the long place line must not rescue it.
    assert [s.external_id for s in signals] == ["RevFakeLow0001", "RevFakeMid0002",
                                                "RevFakeLow0006"]
    assert signals[0].text.startswith(
        "Lugar en Google Maps: Contables Express SAS (Asesor contable)\n\nLlevo tres meses")
    assert "Apify cost this run about $0.0100" in caplog.text


def test_tiktok_feeds_found_videos_to_the_comments_actor(apify, settings):
    signals = ApifyTikTokCollector(settings).collect(CFG)
    (videos_actor, videos_run), (comments_actor, comments_run) = apify.starts
    assert (videos_actor, comments_actor) == (VIDEOS_ACTOR, COMMENTS_ACTOR)
    assert videos_run["run_input"]["hashtags"] == ["contadorescolombia"]  # "#" stripped
    assert comments_run["run_input"]["postURLs"] == [
        "https://www.tiktok.com/@contadora.ejemplo/video/7400000000000000001",
        "https://www.tiktok.com/@pyme.tips/video/7400000000000000002"]
    assert (videos_run["max_items"], comments_run["max_items"]) == (8, 200)
    assert [s.external_id for s in signals] == ["7400000000000100001", "7400000000000200001"]
    assert signals[0].text.startswith("Video de TikTok: Errores comunes")
    assert signals[1].text.startswith("Yo uso Excel")  # its video has no caption


def test_limit_lowers_the_item_caps(apify, settings):
    ApifyTikTokCollector(settings, limit=5).collect(CFG)
    assert [kwargs["max_items"] for _, kwargs in apify.starts] == [5, 5]


def test_hashtag_without_videos_skips_the_comment_run(apify, settings):
    apify.outputs[VIDEOS_ACTOR] = [{"url": "https://www.tiktok.com/@x/video/1", "error": "gone"}]
    assert ApifyTikTokCollector(settings).collect(CFG) == []
    assert [actor for actor, _ in apify.starts] == [VIDEOS_ACTOR]


def test_facebook_posts_and_comments_with_context(apify, settings):
    signals = ApifyFacebookCollector(settings).collect(CFG)
    assert apify.starts[0][1]["run_input"]["startUrls"] == [
        {"url": "https://www.facebook.com/groups/contadoresejemplo"}]
    assert [s.external_id for s in signals] == [
        "post:5550001", "comment:Y29tbWVudDo1NTUwMDAxXzU1NTAwMDI=",
        "comment:Y29tbWVudDo1NTUwMDAxXzU1NTAwMDM=", "comment:Y29tbWVudDo1NTUwMDEwXzU1NTAwMTE="]
    assert signals[0].text.startswith("Grupo de Facebook: Contadores Colombia Ejemplo\n\n")
    assert signals[1].text.startswith("Comentario a esta publicación de un grupo de Facebook:")
    assert signals[3].text.startswith("Esa foto del formulario")  # parent post has no text


def test_failed_run_keeps_what_it_produced(apify, settings, caplog):
    apify.status = "FAILED"
    apify.message = "max total charge reached, see https://api.apify.com/v2/x?signature=abc"
    assert len(ApifyGmapsCollector(settings).collect(CFG)) == 3
    assert "FAILED (max total charge reached, see [link])" in caplog.text
    assert "signature" not in caplog.text


@pytest.mark.parametrize(("error", "starts"), [
    (FakeApiError(401, "token-not-valid", f"bad token {TOKEN}"), 1),
    (FakeApiError(402, "not-enough-usage-to-run-paid-actor"), 1),
    (FakeApiError(400, "monthly-usage-limit-exceeded"), 1),
    (FakeApiError(400, "invalid-input"), 2),  # only that query is skipped
])
def test_account_errors_stop_the_source_and_input_errors_skip_a_target(
        apify, settings, caplog, error, starts):
    apify.outputs[GMAPS_ACTOR] = error
    cfg = {**CFG, "gmaps_queries": ["software contable Bogotá", "contador Medellín"]}
    assert ApifyGmapsCollector(settings).collect(cfg) == []
    assert len(apify.starts) == starts
    assert TOKEN not in caplog.text


def test_server_errors_are_not_rerun(apify, settings):
    apify.outputs[GMAPS_ACTOR] = FakeApiError(503, "server-error")
    assert ApifyGmapsCollector(settings).collect(CFG) == [] and len(apify.starts) == 1


def test_recollecting_adds_nothing(apify, settings, conn):
    first = db.insert_signals(conn, ApifyFacebookCollector(settings).collect(CFG))
    known = db.external_ids(conn, "apify_facebook")
    second = ApifyFacebookCollector(settings, known_ids=known).collect(CFG)
    assert (first, len(second)) == (4, 0)
