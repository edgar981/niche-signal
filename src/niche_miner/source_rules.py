"""Per-source rules shared by extraction, clustering and scoring:

  source_hint      the short 'fuente' the extraction prompt shows for each text
  source_key       the document a signal came from (one video, app listing, place, post)
  country_rule     Maps and Play reviews are Colombian only with an explicit Colombian cue
  strip_boilerplate  job-post lines (requisitos, horario, salario) left out of clustering input"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

# Review sources whose listings mix countries: never infer "co" from the store or locale.
MULTI_COUNTRY_SOURCES = frozenset({"apify_gmaps", "google_play"})
_CITIES = (
    "bogot[aá]", "medell[ií]n", "cali", "barranquilla", "cartagena", "bucaramanga", "pereira",
    "manizales", "c[uú]cuta", "ibagu[eé]", "santa marta", "villavicencio", "pasto",
    "monter[ií]a", "neiva", "armenia", "popay[aá]n", "valledupar", "tunja", "sincelejo",
    "envigado", "itag[uü][ií]", "soacha", "palmira", "chía", "zipaquir[aá]")
COLOMBIAN_CUE = re.compile(
    r"\b(?:" + "|".join(_CITIES) + r"|colombia\w*|dian|cop|pesos|nit|rut|nequi|pse"
    r"|daviplata|bancolombia|davivienda)\b", re.I)
_BOILERPLATE = re.compile(
    r"^\s*(?:requisitos?|requerimientos|perfil requerido|experiencia(?: requerida)?|educaci[oó]n"
    r"|horarios?|jornada|turnos?|salario|sueldo|remuneraci[oó]n|beneficios|tipo de contrato"
    r"|contrato\b|tiempo completo|medio tiempo|\$\s*[\d.,]+)", re.I)


def source_hint(source: str, url: str | None) -> str:
    """Short 'fuente' for the model, e.g. which app a review is about."""
    if source == "google_play" and url:
        app_id = parse_qs(urlparse(url).query).get("id", ["?"])[0]
        return (f"reseña de 1-2 estrellas en Google Play (la tienda mezcla países), app con id "
                f"{app_id}")
    if source == "computrabajo":
        return "oferta de empleo en Computrabajo Colombia, escrita por la empresa que contrata"
    if source == "youtube":
        return ("comentario en un video de YouTube; la línea 'Video:' es el título del video, "
                "contexto que no escribió quien comenta")
    if source == "apify_gmaps":
        return ("reseña de pocas estrellas en Google Maps (el negocio puede atender a varios "
                "países); la línea 'Lugar en Google Maps:' es el negocio reseñado, contexto que "
                "no escribió quien reseña")
    if source == "apify_tiktok":
        return ("comentario en un video de TikTok; la línea 'Video de TikTok:' es la descripción "
                "del video, contexto que no escribió quien comenta")
    if source == "apify_facebook":
        return ("publicación o comentario en un grupo público de Facebook; la primera línea "
                "(nombre del grupo o publicación comentada) es contexto, no de quien escribe")
    return source


def source_key(source: str, url: str | None, signal_id: int) -> str:
    """The document a signal belongs to: a YouTube video (not the comment), a Play listing
    (not the review), a Maps place, a TikTok video, a job post. No URL: the signal itself."""
    if not url:
        return f"{source}:signal:{signal_id}"
    parts = urlparse(url)
    query = parse_qs(parts.query)
    if source == "youtube" and query.get("v"):
        return f"https://www.youtube.com/watch?v={query['v'][0]}"
    if source == "google_play" and query.get("id"):
        return f"https://play.google.com/store/apps/details?id={query['id'][0]}"
    return url.split("#", 1)[0]


def has_colombian_cue(text: str) -> bool:
    return bool(COLOMBIAN_CUE.search(text or ""))


def country_rule(source: str, text: str, country: str) -> str:
    """For Maps and Play reviews, "co" needs an explicit Colombian cue in the text (city, DIAN,
    COP, pesos, NIT, RUT, Nequi, PSE, ...); otherwise it becomes "unknown". Other countries and
    other sources are left as extracted."""
    if source in MULTI_COUNTRY_SOURCES and country == "co" and not has_colombian_cue(text):
        return "unknown"
    return country


def strip_boilerplate(text: str) -> str:
    """Drop job-post boilerplate lines (requisitos, horario, salario, contrato...) from text
    that goes into embedding and theme assignment. Never returns an empty string."""
    kept = [line for line in text.splitlines() if not _BOILERPLATE.match(line)]
    return "\n".join(kept).strip() or text.strip()
