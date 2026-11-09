"""Markdown report (data/reports/YYYY-MM-DD.md): top clusters with score breakdown, quotes,
who / current_tool mix, and a blank hypothesis line per cluster."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from . import db
from .score import formula_markdown
from .settings import Settings

HYPOTHESIS_TEMPLATE = (
    "[Who] pays [$X] for [outcome] because [current alternative] fails at [specific thing]."
)
QUOTE_CHARS = 280


def write_report(conn: sqlite3.Connection, settings: Settings, reports_dir: Path,
                 niche: str | None = None, top: int | None = None,
                 today: date | None = None) -> Path:
    today = today or date.today()
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{today.isoformat()}.md"
    path.write_text(render(conn, settings, niche, top, today), encoding="utf-8")
    return path


def render(conn: sqlite3.Connection, settings: Settings, niche: str | None = None,
           top: int | None = None, today: date | None = None) -> str:
    top = top or settings.report.top_n
    clusters = db.latest_clusters(conn, niche)
    absorbed: dict[int, list[int]] = defaultdict(list)
    for c in clusters:
        if c["merged_into"]:
            absorbed[c["merged_into"]].append(c["id"])
    ranked = sorted(
        (c for c in clusters if not c["is_noise"] and not c["merged_into"] and c["score"] is not None),
        key=lambda c: (c["score"], _breakdown(c).get("uncapped", 0)), reverse=True)

    out = [f"# Niche signal report — {(today or date.today()).isoformat()}", ""]
    out += _overview(conn, clusters)
    out += ["## Scoring formula", "", formula_markdown(settings.scoring), "",
            f"Hypothesis template: _{HYPOTHESIS_TEMPLATE}_", ""]
    out += [f"## Top {min(top, len(ranked))} clusters", ""]
    if not ranked:
        out += ["_No scored clusters yet. Need at least "
                f"{settings.clustering.min_cluster_size} similar complaint pains per cluster; "
                "collect more signals, then run `extract`, `cluster` and `score`._", ""]
    for rank, c in enumerate(ranked[:top], 1):
        out += _cluster_section(conn, rank, c, absorbed.get(c["id"], []))
    out += _all_clusters(ranked, clusters)
    out += _review_stub(ranked[:top])
    return "\n".join(out).rstrip() + "\n"


def _breakdown(c: sqlite3.Row) -> dict:
    return json.loads(c["score_breakdown"]) if c["score_breakdown"] else {}


def _counts(rows: Iterable[sqlite3.Row]) -> str:
    return " · ".join(f"{r['value']} {r['n']}" for r in rows) or "none"


def _overview(conn: sqlite3.Connection, clusters: list[sqlite3.Row]) -> list[str]:
    by_niche: dict[str, list[str]] = defaultdict(list)
    total = 0
    for row in db.signal_counts(conn):
        by_niche[row["niche_tag"]].append(f"{row['source']} {row['n']}")
        total += row["n"]
    sources = "; ".join(f"{n}: {', '.join(parts)}" for n, parts in by_niche.items()) or "none"
    ex = db.extraction_counts(conn)
    runs = sorted({c["run_id"] for c in clusters})
    methods = ", ".join(sorted({c["method"] or "hdbscan" for c in clusters})) or "—"
    n_clusters = sum(1 for c in clusters if not c["is_noise"] and not c["merged_into"])
    noise = sum(len(db.member_details(conn, [c["id"]])) for c in clusters if c["is_noise"])
    countries = [r for r in db.complaint_breakdown(conn, "country")
                 if r["value"] not in ("co", "unknown")]
    lines = ["## Overview", "",
             f"- Signals: {total} ({sources})",
             f"- Extracted: {ex['total']} — {ex['complaints']} complaints, "
             f"{ex['not_complaints']} not complaints, {ex['failed']} failed",
             f"- Pain types: {_counts(db.complaint_breakdown(conn, 'pain_type'))}",
             f"- Complaints from outside Colombia (kept, not scored): {ex['excluded']}"
             + (f" ({_counts(countries)})" if countries else ""),
             f"- Clusters (run {', '.join(runs) or '—'}, method {methods}): {n_clusters} "
             f"clusters, {noise} complaint pains left as noise"]
    usage = {u["purpose"]: u for u in db.llm_usage(conn)}
    if usage:
        cost = sum(u["cost_usd"] for u in usage.values())
        models = ", ".join(sorted({m for u in usage.values() for m in (u["models"] or "").split(",")}))
        line = f"- LLM spend to date: ${cost:.4f} ({models})"
        if "extract" in usage:  # unit cost from the latest extraction run (re-runs don't skew it)
            last = {u["purpose"]: u for u in db.llm_usage(conn, usage["extract"]["last_run"])}
            ext = last["extract"]
            line += (f" — latest extraction: ${ext['cost_usd']:.4f} for {ext['items']} signals ≈ "
                     f"**${1000 * ext['cost_usd'] / max(ext['items'], 1):.2f} per 1,000 signals**")
        if any(u["unpriced"] for u in usage.values()):
            line += " — some calls unpriced (add the model to llm.pricing)"
        lines.append(line)
    return lines + [""]


def _distribution(values: Iterable[str | None], k: int = 6) -> str:
    counts = Counter((v or "").strip().casefold() or "(sin dato)" for v in values)
    parts = [f"{value} ({n})" for value, n in counts.most_common(k)]
    rest = sum(counts.values()) - sum(n for _, n in counts.most_common(k))
    return " · ".join(parts) + (f" · otros ({rest})" if rest else "")


def _quote(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= QUOTE_CHARS else text[: QUOTE_CHARS - 1].rstrip() + "…"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _representatives(c: sqlite3.Row, scored: list[sqlite3.Row], k: int = 3) -> list[sqlite3.Row]:
    """The cluster's stored representatives that are in scope, topped up with the most central
    scored members from other authors (members arrive sorted by distance)."""
    by_pain = {m["pain_id"]: m for m in scored}
    picked = [by_pain[p] for p in json.loads(c["representative_ids"]) if p in by_pain][:k]
    authors = {m["author_hash"] for m in picked}
    for m in scored:
        if len(picked) >= k:
            break
        if m["author_hash"] not in authors:
            picked.append(m)
            authors.add(m["author_hash"])
    return picked


def _cluster_section(conn: sqlite3.Connection, rank: int, c: sqlite3.Row,
                     absorbed: list[int]) -> list[str]:
    members = db.member_details(conn, [c["id"], *absorbed])
    scored = [m for m in members if not m["exclude_from_scoring"]]
    b = _breakdown(c)
    size = c["size"] or 1
    business = round(c["b2b_ratio"] * c["size"])
    meta = (f"cluster `{c['id']}` · niche `{c['niche_tag']}` · {c['size']} pains · "
            f"{c['distinct_authors']} distinct authors")
    if excluded := len(members) - len(scored):
        meta += f" · {excluded} more from outside Colombia, not scored"
    if absorbed:
        meta += f" · merged from {', '.join(map(str, absorbed))}"
    status = "reviewed" if b.get("reviewed") else "not reviewed"
    types = " · ".join(f"{t} {n}" for t, n in Counter(m["pain_type"] or "unknown"
                                                       for m in scored).most_common())
    total_note = f"evidence {b.get('evidence', 0):.1f}"
    if b.get("capped"):
        total_note += f"; capped (reachability = 0; uncapped {b['uncapped']:.1f})"
    lines = [f"### {rank}. {c['label']} — {c['score']:.1f}", "", meta, "",
             "| Component | Input | Points |", "|---|---|---:|",
             f"| Frequency | {c['distinct_authors']} distinct authors | +{b.get('frequency', 0):.1f} |",
             f"| Spend | {c['spend_count']} of {size} pains with a spend signal "
             f"| +{b.get('spend', 0):.1f} |",
             f"| B2B | {business} of {size} pains from businesses | +{b.get('b2b', 0):.1f} |",
             f"| Pain-type weight | {types} | ×{b.get('type_weight', 1):.2f} |",
             f"| Reachability | {c['reachability']:g}/10 ({status}) "
             f"| +{b.get('reachability', 0):.1f} |",
             f"| Competition | {c['competition']:g}/10 ({status}) "
             f"| −{b.get('competition', 0):.1f} |",
             f"| **Score** | {total_note} | **{c['score']:.1f}** |", "",
             "**Representative quotes**", ""]
    for i, m in enumerate(_representatives(c, scored), 1):
        link = f"[{m['source']}]({m['url']})" if m["url"] else m["source"]
        lines += [f"{i}. “{_quote(m['text'])}” — {link}", f"   - _Dolor:_ {m['pain']}"]
    lines += ["", f"**Who:** {_distribution(m['who'] for m in scored)}", "",
              f"**Current tool:** {_distribution(m['current_tool'] for m in scored)}", ""]
    spend = [m["spend_signal"] for m in scored if m["spend_signal"]]
    if spend:
        lines += ["**Spend signals:** " + " · ".join(f"“{_quote(s)}”" for s in spend[:3]), ""]
    return lines + ["**Hypothesis:** ", "", "---", ""]


def _all_clusters(ranked: list[sqlite3.Row], clusters: list[sqlite3.Row]) -> list[str]:
    if not clusters:
        return []
    lines = ["## All clusters in the latest run", "",
             "| id | niche | label | pains | authors | spend | b2b | score (uncapped) |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for c in ranked:
        b = _breakdown(c)
        lines.append(f"| {c['id']} | {c['niche_tag']} | {_cell(c['label'])} | {c['size']} | "
                     f"{c['distinct_authors']} | {c['spend_count']} | {c['b2b_ratio']:.0%} | "
                     f"{c['score']:.1f} ({b.get('uncapped', c['score']):.1f}) |")
    for c in clusters:
        if c["merged_into"]:
            note = f"merged into {c['merged_into']}"
        elif c["is_noise"]:
            note = "noise, not scored"
        elif c["score"] is None:
            note = "not scored (no pains from Colombia)"
        else:
            continue
        lines.append(f"| {c['id']} | {c['niche_tag']} | _{_cell(c['label'])}_ | {c['size']} | "
                     f"{c['distinct_authors']} | {c['spend_count']} | | {note} |")
    return lines + [""]


def _review_stub(top: list[sqlite3.Row]) -> list[str]:
    if not top:
        return []
    lines = ["## Review", "",
             "Fill in and paste into `config/reachability.yaml`, add merges to "
             "`config/merges.yaml` (`target_id: [ids]`), then run "
             "`niche-miner score && niche-miner report`.", "", "```yaml", "clusters:"]
    for c in top:
        lines.append(f"  {c['id']}: {{reachability: {c['reachability']:g}, "
                     f"competition: {c['competition']:g}}}  # {c['label']}")
    return lines + ["```", ""]
