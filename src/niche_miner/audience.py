"""Niche-level reachability_proxy (0-10), scored by channel TYPE, not by instance.

Types and their inputs (per niche):
  job_board           Computrabajo posts in the last 30 days for the niche's own
                      computrabajo_queries (job titles), in the configured cities
  fb_groups           the niche's facebook_groups; member counts from facebook_group_members
  gremio              gremios / asociaciones found by a Google search "<niche> asociación Colombia"
  whatsapp_community  whatsapp_communities listed by hand in niches.yaml
  creator_comments    YouTube / TikTok creators whose videos drew the niche's comments; never a
                      type on its own (we can comment there, not reach anyone directly)

Score: 0 = no type with signal; 3 = one type; 5 = two; 7 = three (+2 per further type); +1 per
type with strong volume (job posts, FB members, a gremio with a public member directory);
creator_comments adds at most +2. Capped at 10.

A cluster's reachability is the niche proxy × who_match: the share of its pains whose `who`
matches one of the niche's buyer_roles (word prefixes, accents ignored), with an unknown `who`
counting as `unknown_who_weight`."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import parse_qs, urlparse

from .extract import Completer
from .llm import LLMError

TYPES = ("job_board", "fb_groups", "gremio", "whatsapp_community")
_H1 = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_COUNT = re.compile(r"([\d.]+)\s+ofertas", re.I)
_TIKTOK_HANDLE = re.compile(r"/@([^/?#]+)")

GREMIO_SYSTEM = """\
You read Google results for a search about Colombian trade associations (gremios, \
asociaciones, federaciones, colegios, cámaras) for one niche. List only organizations that \
appear in the results and represent the people or businesses of that niche in Colombia. For \
each: name, url, whether the results show a public member directory (a list or directory of \
afiliados / asociados / miembros anyone can browse), and the result that shows it. Return an \
empty list if none fits."""
GREMIO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"gremios": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "url": {"type": "string"},
                       "member_directory": {"type": "boolean"}, "evidence": {"type": "string"}},
        "required": ["name", "url", "member_directory", "evidence"],
        "additionalProperties": False}}},
    "required": ["gremios"], "additionalProperties": False,
}


def _fold(text: str) -> str:
    """Lower case without accents: 'Dueña de Ferretería' -> 'duena de ferreteria'."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def matches_role(who: str, roles: Sequence[str]) -> bool:
    """True when a buyer role starts a word of `who` ('empresa' matches 'empresario')."""
    folded = _fold(who)
    return any(re.search(rf"\b{re.escape(_fold(role).strip())}", folded) for role in roles
               if role.strip())


def who_match(members: Iterable[Mapping[str, Any]], roles: Sequence[str],
              unknown_weight: float) -> tuple[float, dict[str, int]]:
    """(share of pains from buyers, counts). No buyer_roles configured: 1.0 (no discount)."""
    counts = {"buyer": 0, "unknown": 0, "other": 0}
    for m in members:
        who = (m["who"] or "").strip()
        key = "unknown" if not who else "buyer" if matches_role(who, roles) else "other"
        counts[key] += 1
    total = sum(counts.values())
    if not roles or not total:
        return 1.0, counts
    return (counts["buyer"] + unknown_weight * counts["unknown"]) / total, counts


def parse_offer_count(page_html: str) -> int:
    """'585 Ofertas de trabajo de auxiliar contable en Bogotá, D.C.' -> 585; no number -> 0."""
    for m in _H1.finditer(page_html):
        if found := _COUNT.search(" ".join(_TAG.sub(" ", m.group(1)).split())):
            return int(found.group(1).replace(".", ""))
    return 0


def youtube_video_ids(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    ids = {parse_qs(urlparse(r["url"] or "").query).get("v", [""])[0]
           for r in rows if r["source"] == "youtube"}
    return sorted(i for i in ids if i)


def tiktok_accounts(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    handles = {h.group(1) for r in rows if r["source"] == "apify_tiktok"
               if (h := _TIKTOK_HANDLE.search(r["url"] or ""))}
    return sorted(handles)


def find_gremios(llm: Completer, model: str, niche: str, results: list[dict]) -> list[dict]:
    """Gremios named in the Google results, from one Claude call. Raises LLMError."""
    if not results:
        return []
    user = json.dumps({"niche": niche, "google_results": results}, ensure_ascii=False)
    text = llm.complete(purpose="review", model=model, system=GREMIO_SYSTEM, user=user,
                        schema=GREMIO_SCHEMA, n_items=1)
    try:
        found = json.loads(text)["gremios"]
    except (ValueError, KeyError, TypeError) as e:
        raise LLMError(f"gremio response is not valid JSON ({e})") from e
    return [g for g in found if isinstance(g, dict) and g.get("name")]


def _k(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def niche_score(types: Mapping[str, Mapping[str, Any]], creator: Mapping[str, Any]
                ) -> tuple[int, str]:
    """(0-10, why) from {type: {"signal", "strong", "summary"}} and creator {"points", ...}."""
    with_signal = [t for t in TYPES if types.get(t, {}).get("signal")]
    strong = [t for t in with_signal if types[t].get("strong")]
    base = 0 if not with_signal else 3 + 2 * (len(with_signal) - 1)
    creator_pts = min(2, int(creator.get("points") or 0))
    score = min(10, base + len(strong) + creator_pts)
    parts = [f"{len(with_signal)} channel type(s) with signal → {base}"]
    if strong:
        parts.append(f"+{len(strong)} strong ({', '.join(strong)})")
    if creator_pts:
        parts.append(f"+{creator_pts} creator comments")
    return score, " ".join(parts) + f" = {score}"


def creator_points(subscribers: Sequence[int | None], tiktok: Sequence[str],
                   strong_audience: int) -> int:
    """+1 when creators' videos drew the niche's comments, +2 when their known audience is
    at least `strong_audience` combined."""
    if not subscribers and not tiktok:
        return 0
    return 2 if sum(s or 0 for s in subscribers) >= strong_audience else 1


def summary_lines(review: Mapping[str, Any]) -> list[str]:
    """One line per type for the report."""
    types = review.get("types") or {}
    lines = []
    job = types.get("job_board") or {}
    lines.append(f"job_board: {_k(job.get('posts', 0))} Computrabajo posts in 30 days "
                 f"({', '.join(job.get('queries') or []) or 'no computrabajo_queries'}; "
                 f"{', '.join(job.get('cities') or [])})" + (" — strong" if job.get("strong") else ""))
    fb = types.get("fb_groups") or {}
    members = fb.get("members")
    lines.append(f"fb_groups: {len(fb.get('groups') or [])} group(s) listed"
                 + (f", {_k(members)} members" if members else ", member counts not available")
                 + (" — strong" if fb.get("strong") else ""))
    gr = types.get("gremio") or {}
    names = ", ".join(g["name"] + (" (public directory)" if g.get("member_directory") else "")
                      for g in gr.get("gremios") or [])
    lines.append(f"gremio: {len(gr.get('gremios') or [])} found for “{gr.get('query', '')}”"
                 + (f": {names}" if names else "") + (" — strong" if gr.get("strong") else ""))
    wa = types.get("whatsapp_community") or {}
    lines.append(f"whatsapp_community: {len(wa.get('communities') or [])} listed")
    cr = review.get("creator_comments") or {}
    chans = ", ".join(f"{c['name']} ({_k(c['audience']) if c.get('audience') else 'n/a'})"
                      for c in cr.get("channels") or [])
    lines.append(f"creator_comments (+{cr.get('points', 0)}, never a type): {chans or 'none'}")
    return lines
