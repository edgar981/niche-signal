"""Computrabajo Colombia job posts: keyword search pages, then each offer's JSON-LD JobPosting.

Computrabajo's legal terms forbid scripts and crawlers; the owner chose to run this at low
volume anyway (2026-11-09). So it is polite by construction: one request every min_interval_s
(at least 2s), a desktop browser User-Agent, every URL checked against robots.txt, offers
already in the DB never re-fetched, small per-query caps, and HTTP 403/429 stops the source.

The hiring company is only hashed (its Computrabajo id). Emails, phone numbers and links are
scrubbed from the text, and the company name is left out of it."""

from __future__ import annotations

import html
import json
import logging
import re
import unicodedata
from collections.abc import Collection, Iterator
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ..models import RawSignal
from ..settings import Settings
from .base import Collector, SkipTarget, StopSource, hash_author
from .robots import Robots

log = logging.getLogger(__name__)

SOURCE = "computrabajo"
_OFFER_HREF = re.compile(
    r'href="(/ofertas-de-trabajo/oferta-de-trabajo-de-[^"#?]+-([0-9A-F]{32}))')
_LD_JSON = re.compile(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', re.S)
_H1 = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S)
_H2 = re.compile(r"<h2[^>]*>.*?</h2>", re.S)
_OFFER_BLOCK = re.compile(
    r'div-link="oferta"[^>]*>(.*?)(?:<p[^>]*>\s*Aptitudes asociadas|<div[^>]*posSticky|div-link=)',
    re.S)
_BREAK = re.compile(r"<br\s*/?>|</(?:p|li|div|h\d)>", re.I)
_TAG = re.compile(r"<[^>]+>")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_LINK = re.compile(r"(?:https?://|www\.)\S+", re.I)
# Colombian mobiles (3xx) and landlines (60x), 10 digits, optional +57 — not salary amounts.
_PHONE = re.compile(r"(?<!\d)(?:\+?57[\s-]?)?\(?(?:3\d{2}|60\d)\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_UNITS = {"HOUR": "por hora", "DAY": "diario", "WEEK": "semanal", "MONTH": "mensual",
          "YEAR": "anual"}
ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"  # a browser's default


class PageGone(SkipTarget):
    """HTTP 404/410: the page doesn't exist (an expired offer, or no robots.txt)."""


def search_url(base_url: str, query: str, page: int = 1) -> str:
    """'conciliación bancaria' -> {base}/trabajo-de-conciliacion-bancaria[?p=N]."""
    ascii_query = unicodedata.normalize("NFKD", query.casefold()).encode("ascii", "ignore")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_query.decode()).strip("-")
    url = f"{base_url.rstrip('/')}/trabajo-de-{slug}"
    return url if page == 1 else f"{url}?p={page}"


def parse_search(page_html: str, base_url: str, page: int) -> tuple[list[tuple[str, str]], bool]:
    """(offer id, offer URL) pairs in page order, and whether the page links to the next one."""
    offers: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path, offer_id in _OFFER_HREF.findall(page_html):
        if offer_id not in seen:
            seen.add(offer_id)
            offers.append((offer_id, base_url.rstrip("/") + path))
    has_next = re.search(rf'data-path="[^"]*[?&]p={page + 1}"', page_html) is not None
    return offers, has_next


def html_to_text(fragment: str) -> str:
    text = _TAG.sub(" ", _BREAK.sub("\n", html.unescape(fragment)))
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def scrub(text: str) -> str:
    """Drop contact details: job posts often ask to send a CV to an email or WhatsApp."""
    return _PHONE.sub("[teléfono]", _LINK.sub("[enlace]", _EMAIL.sub("[correo]", text)))


def job_posting(page_html: str) -> dict[str, Any] | None:
    for block in _LD_JSON.findall(page_html):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        items = data.get("@graph", [data]) if isinstance(data, dict) else data
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and item.get("@type") == "JobPosting":
                return item
    return None


def _money(amount: Any) -> str | None:
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return None
    return f"$ {value:,.0f}".replace(",", ".") if value > 0 else None


def _salary(posting: dict[str, Any]) -> str | None:
    salary = posting.get("baseSalary")
    if not isinstance(salary, dict):
        return None
    value = salary.get("value")
    if isinstance(value, dict):
        amounts = [_money(value.get(k)) for k in ("value", "minValue", "maxValue")]
        unit = _UNITS.get(str(value.get("unitText") or "").upper())
    else:
        amounts, unit = [_money(value)], None
    shown = list(dict.fromkeys(a for a in amounts if a))
    if not shown:
        return None
    currency = salary.get("currency") or posting.get("salaryCurrency")
    return " ".join(p for p in (" a ".join(shown[:2]), currency, unit) if p)


def _location(posting: dict[str, Any]) -> str | None:
    place = posting.get("jobLocation")
    if isinstance(place, list):
        place = place[0] if place else None
    address = place.get("address") if isinstance(place, dict) else None
    if not isinstance(address, dict):
        return None
    parts = (address.get("addressLocality"), address.get("addressRegion"))
    unique = dict.fromkeys(p.strip() for p in parts if isinstance(p, str) and p.strip())
    return ", ".join(unique) or None


def _date(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _company_key(posting: dict[str, Any], offer_id: str) -> str:
    """Computrabajo's company id. Posts without one (e.g. confidential employers) count as
    their own author rather than being lumped together under a shared placeholder name."""
    identifier = posting.get("identifier")
    key = identifier.get("value") if isinstance(identifier, dict) else None
    return f"company:{key.strip()}" if isinstance(key, str) and key.strip() else f"offer:{offer_id}"


def _html_offer(page_html: str) -> tuple[str, str] | None:
    """Fallback for offers without JSON-LD (confidential employers): the <h1> title and the
    'Descripción de la oferta' block (salary/contract tags, description, Requerimientos)."""
    title, block = _H1.search(page_html), _OFFER_BLOCK.search(page_html)
    if not title or not block:
        return None
    return html_to_text(title.group(1)), html_to_text(_H2.sub("", block.group(1), count=1))


def parse_offer(page_html: str, offer_id: str, url: str, niche_tag: str) -> RawSignal | None:
    """Title, sector, place and salary as header lines, then the description (which ends with
    Computrabajo's 'Requerimientos' list: education, experience, required software)."""
    posting = job_posting(page_html) or {}
    if posting:
        title = html_to_text(str(posting.get("title") or ""))
        description = html_to_text(str(posting.get("description") or ""))
        header = (("Sector", posting.get("industry")), ("Ubicación", _location(posting)),
                  ("Salario", _salary(posting)))
    elif fallback := _html_offer(page_html):
        (title, description), header = fallback, ()
    else:
        return None
    if not description:
        return None
    lines = [title]
    for label, value in header:
        if isinstance(value, str) and value.strip():
            lines.append(f"{label}: {value.strip()}")
    return RawSignal(
        niche_tag=niche_tag,
        source=SOURCE,
        external_id=offer_id,
        url=url,
        author_hash=hash_author(SOURCE, _company_key(posting, offer_id)),
        author_type="business",  # the writer is the employer
        text=scrub("\n".join(line for line in lines if line) + "\n\n" + description),
        posted_at=_date(posting.get("datePosted")),
    )


class ComputrabajoCollector(Collector):
    source = SOURCE
    config_key = "computrabajo_queries"

    def __init__(self, settings: Settings, limit: int | None = None,
                 known_ids: Collection[str] = frozenset()):
        super().__init__(settings, limit, known_ids)
        self.cfg = settings.collectors.computrabajo
        self.min_interval_s = self.cfg.min_interval_s
        self.robots: Robots | None = None
        self._requested: set[str] = set()  # offers fetched this run, across queries

    def fetch(self, query: str, niche_tag: str) -> Iterator[RawSignal]:
        if self.robots is None:
            self.robots = Robots(self.call(self._robots_txt))
        fetched = 0
        for page in range(1, self.cfg.max_pages_per_query + 1):
            offers, has_next = parse_search(self.call(self._get, search_url(
                self.cfg.base_url, query, page)), self.cfg.base_url, page)
            if page == 1 and not offers:
                raise SkipTarget("no offers on the search page (query or page layout changed?)")
            for offer_id, url in offers:
                if offer_id in self.known_ids or offer_id in self._requested:
                    continue
                if fetched >= self.cfg.max_offers_per_query:
                    return
                self._requested.add(offer_id)
                fetched += 1
                try:
                    page_html = self.call(self._get, url)
                except SkipTarget as e:  # expired or removed offer
                    log.info("%s: %s", SOURCE, e)
                    continue
                if signal := parse_offer(page_html, offer_id, url, niche_tag):
                    yield signal
                else:
                    log.warning("%s: no JobPosting data in %s", SOURCE, url)
            if not has_next:
                return

    def _robots_txt(self) -> str:
        try:
            return self._get(f"{self.cfg.base_url.rstrip('/')}/robots.txt")
        except PageGone:  # no robots.txt: nothing is disallowed
            return ""
        except SkipTarget as e:  # any other refusal: don't guess what is allowed
            raise StopSource(f"can't read robots.txt ({e})") from e

    def _get(self, url: str) -> str:
        """One page as text. 403/429 stop the source, other 4xx skip the page without
        retrying, 5xx and network errors are retried with backoff by call()."""
        if self.robots is not None and not self.robots.allowed(url):
            raise SkipTarget(f"robots.txt disallows {url}")
        request = Request(url, headers={"User-Agent": self.cfg.user_agent, "Accept": ACCEPT,
                                        "Accept-Language": "es-CO,es;q=0.9"})
        try:
            with urlopen(request, timeout=self.cfg.timeout_s) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except HTTPError as e:
            if e.code in (403, 429):
                raise StopSource(f"HTTP {e.code} on {url}") from e
            if e.code in (404, 410):
                raise PageGone(f"HTTP {e.code} on {url}") from e
            if 400 <= e.code < 500:
                raise SkipTarget(f"HTTP {e.code} on {url}") from e
            raise
