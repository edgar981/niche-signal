"""Computrabajo parsers on saved HTML fixtures, robots.txt rules, and the collector over a fake
web server (no network)."""

import io
import json
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest
from pydantic import ValidationError

from niche_miner import db
from niche_miner.collectors import computrabajo
from niche_miner.collectors.base import hash_author
from niche_miner.collectors.computrabajo import (
    ComputrabajoCollector,
    job_posting,
    parse_offer,
    parse_search,
    search_url,
)
from niche_miner.collectors.pii import scrub
from niche_miner.collectors.robots import Robots
from niche_miner.settings import ComputrabajoSettings, Settings

FIXTURES = Path(__file__).parent / "fixtures"
SEARCH = (FIXTURES / "computrabajo_search.html").read_text("utf-8")
OFFER = (FIXTURES / "computrabajo_offer.html").read_text("utf-8")
BASE = "https://co.computrabajo.com"
ID_A, ID_B, ID_C, ID_D = ("1111AAAA2222BBBB3333CCCC4444DDDD", "5555EEEE6666FFFF7777AAAA8888BBBB",
                          "9999CCCC0000DDDD1111EEEE2222FFFF", "0000BBBB1111CCCC2222DDDD3333EEEE")
OFFER_A_URL = (f"{BASE}/ofertas-de-trabajo/oferta-de-trabajo-de-auxiliar-contable-y-"
               f"administrativo-en-bogota-dc-{ID_A}")
EXPECTED_TEXT = """\
Auxiliar contable y administrativo - Excel & conciliaciones
Sector: Venta al por mayor
Ubicación: Bogotá, D.C.
Salario: $ 1.750.905 COP mensual

Distribuidora de insumos para panaderías requiere auxiliar contable.
Registrar las facturas de proveedores en Siigo y hacer las conciliaciones bancarias de cada mes.
Envía tu hoja de vida a [correo] o al WhatsApp [teléfono]. Más información en [enlace]
-Requerimientos- Educación mínima: Técnico
1 año de experiencia
Conocimientos: Conciliaciones bancarias, Microsoft excel, Siigo"""


def test_search_url_slugs_the_query():
    assert search_url(BASE, "Conciliación bancaria") == f"{BASE}/trabajo-de-conciliacion-bancaria"
    assert search_url(BASE + "/", "auxiliar contable excel", 2) == (
        f"{BASE}/trabajo-de-auxiliar-contable-excel?p=2")


def test_parse_search_lists_offers_once_in_page_order():
    offers, has_next = parse_search(SEARCH, BASE, page=1)
    assert [offer_id for offer_id, _ in offers] == [ID_C, ID_A, ID_B]
    assert offers[1][1] == OFFER_A_URL  # absolute, without the #lc= tracking fragment
    assert has_next
    assert parse_search(SEARCH, BASE, page=2)[1] is False  # no link to page 3


def test_parse_offer_builds_text_from_job_posting():
    signal = parse_offer(OFFER, ID_A, OFFER_A_URL, "contadores_pymes")
    assert signal.text == EXPECTED_TEXT
    assert (signal.source, signal.external_id, signal.url) == ("computrabajo", ID_A, OFFER_A_URL)
    assert signal.niche_tag == "contadores_pymes"
    assert signal.author_type == "business"
    assert signal.author_hash == hash_author("computrabajo", "company:0123456789ABCDEF")
    assert signal.posted_at == datetime(2026, 9, 22, tzinfo=UTC)


def test_contacts_and_company_name_are_never_stored():
    dumped = parse_offer(OFFER, ID_A, OFFER_A_URL, "n").model_dump_json()
    for secret in ("seleccion@", "310 555 0123", "https://distribuidora", "Ejemplo SAS"):
        assert secret not in dumped


def test_offer_without_company_id_is_its_own_author():
    posting = job_posting(OFFER)
    del posting["identifier"]
    signal = parse_offer(ld_page(posting), ID_A, OFFER_A_URL, "n")
    assert signal.author_hash == hash_author("computrabajo", f"offer:{ID_A}")


def test_confidential_offer_is_read_from_the_html():
    page = (FIXTURES / "computrabajo_offer_confidential.html").read_text("utf-8")
    signal = parse_offer(page, ID_B, offer_url(ID_B), "n")
    assert signal.text == (
        "Analista contable - Chapinero\n\n$ 2.764.130,00 (Mensual)\nTiempo Completo\n"
        "Importante empresa del sector requiere analista contable para los cierres mensuales y "
        "las conciliaciones bancarias.\nHojas de vida al [teléfono].\nRequerimientos\n"
        "Educación mínima: Universidad / Carrera Profesional\n"
        "Conocimientos: Conciliaciones bancarias, Microsoft excel")
    assert signal.author_hash == hash_author("computrabajo", f"offer:{ID_B}")
    assert signal.author_type == "business" and signal.posted_at is None


def test_page_without_offer_data_is_skipped():
    assert parse_offer(SEARCH, ID_A, OFFER_A_URL, "n") is None


def test_scrub_keeps_salaries_and_drops_contacts():
    text = "Pago $ 1.750.905 o 2500000 al mes. Cel +57 310-555-0123, fijo (601) 555 0123, rh@x.co"
    assert scrub(text) == (
        "Pago $ 1.750.905 o 2500000 al mes. Cel [teléfono], fijo [teléfono], [correo]")


ROBOTS = """\
User-agent: Googlebot
Disallow: /

User-agent: *
Disallow: /ofertas-de-trabajo/*pubdate=
Disallow: /Ajax/*
Allow: /Ajax/public
Disallow: /private$
"""


def test_robots_rules_with_wildcards():
    robots = Robots(ROBOTS)
    assert robots.allowed(f"{BASE}/trabajo-de-auxiliar-contable?p=2")  # Googlebot group ignored
    assert robots.allowed(OFFER_A_URL)
    assert not robots.allowed(f"{BASE}/ofertas-de-trabajo/?q=contador&pubdate=1")
    assert not robots.allowed(f"{BASE}/Ajax/offers")
    assert robots.allowed(f"{BASE}/Ajax/public")  # the longer Allow wins
    assert not robots.allowed(f"{BASE}/private") and robots.allowed(f"{BASE}/private/x")
    assert Robots("").allowed(f"{BASE}/anything")


def test_polite_settings_are_enforced():
    with pytest.raises(ValidationError):
        ComputrabajoSettings(min_interval_s=1.0)


# --- collector over a fake web server ---------------------------------------------------------

def ld_page(posting: dict) -> str:
    return f'<script type="application/ld+json">{json.dumps(posting)}</script>'


def offer_url(offer_id: str) -> str:
    return f"{BASE}/ofertas-de-trabajo/oferta-de-trabajo-de-cargo-{offer_id}"


def offer_page(offer_id: str) -> str:
    posting = job_posting(OFFER)
    posting["description"] = (f"Empresa requiere persona para la oferta {offer_id}: registrar "
                              "facturas y conciliar los bancos cada semana en Excel.")
    return ld_page(posting)


def page_one() -> str:
    """The fixture search page, with offer links pointing at offer_url()."""
    html = SEARCH
    for offer_id, url in parse_search(SEARCH, BASE, 1)[0]:
        html = html.replace(url.removeprefix(BASE), offer_url(offer_id).removeprefix(BASE))
    return html


class FakeResponse(io.BytesIO):
    def __init__(self, body: str):
        super().__init__(body.encode("utf-8"))
        self.headers = Message()
        self.headers["Content-Type"] = "text/html; charset=utf-8"


@pytest.fixture
def web(monkeypatch):
    """url -> HTML body or HTTP status code; unknown URLs are 404. Records every request."""
    query = f"{BASE}/trabajo-de-auxiliar-contable-excel"
    pages: dict = {
        f"{BASE}/robots.txt": ROBOTS,
        query: page_one(),
        f"{query}?p=2": f'<article><a href="{offer_url(ID_D).removeprefix(BASE)}">x</a></article>',
        **{offer_url(i): offer_page(i) for i in (ID_A, ID_B, ID_C, ID_D)},
    }
    requests: list = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        body = pages.get(request.full_url, 404)
        if isinstance(body, int):
            raise HTTPError(request.full_url, body, "error", Message(), None)
        return FakeResponse(body)

    monkeypatch.setattr(computrabajo, "urlopen", fake_urlopen)
    return pages, requests


@pytest.fixture
def fast():
    settings = Settings()
    settings.collectors.backoff_s = 0
    settings.collectors.computrabajo.min_interval_s = 0
    return settings


CFG = {"tag": "contadores_pymes", "computrabajo_queries": ["auxiliar contable excel"]}


def requested(requests) -> list[str]:
    return [r.full_url.removeprefix(BASE) for r in requests]


def test_collects_new_offers_across_pages(web, fast):
    _, requests = web
    signals = ComputrabajoCollector(fast, known_ids={ID_A}).collect(CFG)
    assert [s.external_id for s in signals] == [ID_C, ID_B, ID_D]
    assert offer_url(ID_A).removeprefix(BASE) not in requested(requests)  # already in the DB
    assert requested(requests)[:2] == ["/robots.txt", "/trabajo-de-auxiliar-contable-excel"]
    ua = fast.collectors.computrabajo.user_agent
    assert all(r.get_header("User-agent") == ua for r in requests)
    assert all(s.lang == "es" for s in signals)


def test_offer_cap_per_query(web, fast):
    _, requests = web
    fast.collectors.computrabajo.max_offers_per_query = 2
    assert len(ComputrabajoCollector(fast).collect(CFG)) == 2
    assert not any(u.endswith("?p=2") for u in requested(requests))


def test_http_403_stops_the_source_and_keeps_what_was_fetched(web, fast):
    pages, requests = web
    pages[offer_url(ID_A)] = 403
    cfg = {**CFG, "computrabajo_queries": ["auxiliar contable excel", "conciliacion bancaria"]}
    signals = ComputrabajoCollector(fast).collect(cfg)
    assert [s.external_id for s in signals] == [ID_C]
    assert requested(requests)[-1] == offer_url(ID_A).removeprefix(BASE)  # nothing after it


def test_missing_offer_is_skipped(web, fast):
    pages, _ = web
    pages[offer_url(ID_C)] = 404
    assert [s.external_id for s in ComputrabajoCollector(fast).collect(CFG)] == [ID_A, ID_B, ID_D]


def test_client_errors_are_not_retried(web, fast):
    pages, requests = web
    pages[offer_url(ID_C)] = 406
    assert [s.external_id for s in ComputrabajoCollector(fast).collect(CFG)] == [ID_A, ID_B, ID_D]
    assert requested(requests).count(offer_url(ID_C).removeprefix(BASE)) == 1


@pytest.mark.parametrize("status", [406, 403])
def test_unreadable_robots_txt_stops_the_source(web, fast, status):
    pages, requests = web
    pages[f"{BASE}/robots.txt"] = status
    assert ComputrabajoCollector(fast).collect(CFG) == []
    assert requested(requests) == ["/robots.txt"]


def test_no_robots_txt_allows_everything(web, fast):
    pages, _ = web
    del pages[f"{BASE}/robots.txt"]  # 404
    assert len(ComputrabajoCollector(fast).collect(CFG)) == 4


def test_robots_disallow_means_no_request(web, fast):
    pages, requests = web
    pages[f"{BASE}/robots.txt"] = "User-agent: *\nDisallow: /trabajo-de-\n"
    assert ComputrabajoCollector(fast).collect(CFG) == []
    assert requested(requests) == ["/robots.txt"]


def test_recollecting_fetches_no_known_offers(web, fast, conn):
    _, requests = web
    first = db.insert_signals(conn, ComputrabajoCollector(fast).collect(CFG))
    requests.clear()
    known = db.external_ids(conn, "computrabajo")
    second = db.insert_signals(conn, ComputrabajoCollector(fast, known_ids=known).collect(CFG))
    assert (first, second) == (4, 0)
    assert all("/ofertas-de-trabajo/" not in u for u in requested(requests))
