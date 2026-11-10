"""Automated competition score (0-10) for one cluster: Google results for three queries built
from the theme label, a Play Store search (country co), the tools the complainers use today,
and one Claude call that lists incumbents and applies the rubric."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .extract import Completer
from .llm import LLMError

log = logging.getLogger(__name__)

SYSTEM = """\
You assess competition for one niche: a specific unmet need of Colombian small and medium \
businesses (SMBs), given as a theme label plus example pains in Spanish. You also get Google \
results for three queries, Play Store search results for Colombia, and the tools the people \
complaining use today (with counts).

List incumbents that solve this specific need for Colombian SMBs; for each: name, country, \
whether it solves the exact need or only an adjacent one, and where you saw it (a result URL, \
"Play Store", "current tool" or "general knowledge"). The tools they use today count as \
incumbents when they do this job, even if the complaints say they do it badly. Then score \
competition 0-10 with this rubric: 0 = nothing specific; 5 = generic tools partly cover it; \
8-10 = a Colombian incumbent does exactly this. Give a two- or three-sentence rationale in \
English. Do not invent products the inputs or well-known facts don't support."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "incumbents": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "country": {"type": "string"},
                           "fit": {"type": "string", "enum": ["exact", "adjacent"]},
                           "evidence": {"type": "string"}},
            "required": ["name", "country", "fit", "evidence"], "additionalProperties": False}},
        "competition": {"type": "integer", "enum": list(range(11))},
        "rationale": {"type": "string"},
    },
    "required": ["incumbents", "competition", "rationale"], "additionalProperties": False,
}


def build_queries(label: str, suffixes: Sequence[str]) -> list[str]:
    return [f"{label} {suffix}".strip() for suffix in suffixes]


def parse_search(items: Iterable[Mapping[str, Any]], per_query: int) -> dict[str, list[dict]]:
    """Organic results per query from the Google search actor's dataset (one item per page)."""
    out: dict[str, list[dict]] = {}
    for item in items:
        term = str((item.get("searchQuery") or {}).get("term") or "").strip()
        if not term:
            continue
        results = out.setdefault(term, [])
        for r in item.get("organicResults") or []:
            if len(results) < per_query and r.get("url"):
                results.append({"title": str(r.get("title") or "")[:200], "url": str(r["url"]),
                                "snippet": str(r.get("description") or "")[:300]})
    return out


def play_search(term: str, hits: int) -> list[dict]:
    """Play Store search for Colombia (google-play-scraper). Errors give an empty list."""
    from google_play_scraper import search

    try:
        found = search(term, lang="es", country="co", n_hits=hits)
    except Exception as e:  # the scraper raises plain exceptions on layout changes
        log.warning("review: Play Store search for %r failed: %s", term, e)
        return []
    keep = ("appId", "title", "developer", "installs", "score", "summary")
    return [{k: a.get(k) for k in keep if a.get(k) is not None} for a in found[:hits]]


def current_tools(members: Iterable[Mapping[str, Any]], k: int = 8) -> list[list]:
    counts = Counter((m["current_tool"] or "").strip() for m in members)
    counts.pop("", None)
    return [[tool, n] for tool, n in counts.most_common(k)]


def assess(llm: Completer, model: str, label: str, pains: Sequence[str],
           search: Mapping[str, list[dict]], play: Sequence[dict],
           tools: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """{'score', 'rationale', 'incumbents'} from one Claude call. Raises LLMError."""
    user = json.dumps({
        "theme_label": label, "example_pains": list(pains),
        "google_results": search or "unavailable (no search this run)",
        "play_store_results_co": list(play), "current_tools": [list(t) for t in tools],
    }, ensure_ascii=False)
    text = llm.complete(purpose="review", model=model, system=SYSTEM, user=user, schema=SCHEMA,
                        n_items=1)
    try:
        data = json.loads(text)
        score = int(data["competition"])
    except (ValueError, KeyError, TypeError) as e:
        raise LLMError(f"competition response is not valid JSON ({e})") from e
    incumbents = [i for i in data.get("incumbents") or [] if isinstance(i, dict) and i.get("name")]
    return {"score": max(0, min(10, score)), "rationale": str(data.get("rationale") or ""),
            "incumbents": incumbents}
