"""Markdown report (data/reports/YYYY-MM-DD.md): niche reachability by channel type, a
top-clusters table, then per cluster the score breakdown, incumbents, who_match, quotes, who /
current_tool mix and a blank hypothesis line; then every cluster, the ones not yet reviewed and
an overrides stub."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from . import db, reviews
from . import report_tables as tables
from .audience import summary_lines
from .report_tables import bd
from .score import formula_markdown
from .settings import Settings

HYPOTHESIS_TEMPLATE = (
    "[Who] pays [$X] for [outcome] because [current alternative] fails at [specific thing]."
)
QUOTE_CHARS = 280


def write_report(conn: sqlite3.Connection, settings: Settings, reports_dir: Path,
                 niche: str | None = None, top: int | None = None,
                 today: date | None = None, review_dir: Path | None = None) -> Path:
    today = today or date.today()
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{today.isoformat()}.md"
    path.write_text(render(conn, settings, niche, top, today, review_dir), encoding="utf-8")
    return path


def render(conn: sqlite3.Connection, settings: Settings, niche: str | None = None,
           top: int | None = None, today: date | None = None,
           review_dir: Path | None = None) -> str:
    top = top or settings.report.top_n
    clusters = db.latest_clusters(conn, niche)
    absorbed: dict[int, list[int]] = defaultdict(list)
    for c in clusters:
        if c["merged_into"]:
            absorbed[c["merged_into"]].append(c["id"])
    ranked = sorted((c for c in clusters if not c["is_noise"] and not c["merged_into"]
                     and c["score"] is not None), key=tables.rank_key, reverse=True)
    active = [c for c in ranked if bd(c).get("status") != "not_a_niche"]
    rk = tables.ranks(active)

    out = [f"# Niche signal report — {(today or date.today()).isoformat()}", ""]
    out += _overview(conn, clusters, active)
    out += ["## Scoring formula", "", formula_markdown(settings.scoring), "",
            f"Hypothesis template: _{HYPOTHESIS_TEMPLATE}_", ""]
    out += _niche_reach(sorted({c["niche_tag"] for c in clusters}), review_dir)
    out += [f"## Top {min(top, len(active))} clusters", ""]
    if not active:
        out += ["_No scored clusters yet. Need at least "
                f"{settings.clustering.min_cluster_size} similar complaint pains per cluster; "
                "collect more signals, then run `extract`, `cluster` and `score`._", ""]
    else:
        if dropped := len(ranked) - len(active):
            out += [f"{dropped} cluster(s) marked `not_a_niche` are left out here and listed in "
                    "the full table.", ""]
        out += tables.top_table(active[:top], rk, settings.report)
    for c in active[:top]:
        out += _cluster_section(conn, c, absorbed.get(c["id"], []), rk, settings)
    out += tables.all_clusters(ranked, clusters, rk)
    out += tables.not_reviewed(ranked)
    out += tables.overrides_stub(ranked)
    return "\n".join(out).rstrip() + "\n"


def _counts(rows: Iterable[sqlite3.Row]) -> str:
    return " · ".join(f"{r['value']} {r['n']}" for r in rows) or "none"


def _overview(conn: sqlite3.Connection, clusters: list[sqlite3.Row],
              active: list[sqlite3.Row]) -> list[str]:
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
    auto = [bd(c)["auto"]["fetched_at"] for c in active if bd(c).get("auto")]
    lines = ["## Overview", "",
             f"- Signals: {total} ({sources})",
             f"- Extracted: {ex['total']} — {ex['complaints']} complaints, "
             f"{ex['not_complaints']} not complaints, {ex['failed']} failed",
             f"- Pain types: {_counts(db.complaint_breakdown(conn, 'pain_type'))}",
             f"- Complaint countries: {_counts(db.complaint_breakdown(conn, 'country'))}",
             f"- Complaints from outside Colombia (kept, not scored): {ex['excluded']}"
             + (f" ({_counts(countries)})" if countries else ""),
             f"- Clusters (run {', '.join(runs) or '—'}, method {methods}): {n_clusters} "
             f"clusters, {noise} complaint pains left as noise",
             f"- Automated review: {len(auto)} of {len(active)} active clusters"
             + (f" (fetched {min(auto)[:10]} to {max(auto)[:10]})" if auto else
                " — run `niche-miner review`")]
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


def _niche_reach(niches: list[str], review_dir: Path | None) -> list[str]:
    lines = ["## Niche reachability", "",
             "Scored by channel type: 0 = no type with signal, 3 = one, 5 = two, 7 = three; +1 "
             "per type with strong volume; creator comments add at most +2 and never count as a "
             "type. A cluster gets the niche value × its who_match.", ""]
    for tag in niches:
        review = reviews.load_niche_auto(review_dir, tag)
        if not review:
            lines += [f"- `{tag}`: no automated review yet — run `niche-miner review`.", ""]
            continue
        lines += [f"**`{tag}` — reachability_proxy {review.get('score')}/10** "
                  f"({review.get('rationale', '')}; fetched {str(review.get('fetched_at'))[:10]})",
                  ""] + [f"- {line}" for line in summary_lines(review)] + [""]
    return lines


def _input(b: dict, part: str) -> str:
    auto, override, value = (b.get(f"{part}_{k}") for k in ("auto", "override", "value"))
    shown = f"{value:g}/10" if value is not None else "—"
    auto_text = f"automated {auto:g}" if auto is not None else "no automated value yet"
    if part == "reachability" and b.get("niche_proxy") is not None:
        auto_text = (f"automated {auto:g} = niche {b['niche_proxy']:g} × who_match "
                     f"{b.get('who_match', 1):.2f}")
    over_text = f"override {override:g}" if override is not None else "no override"
    return f"{shown} ({auto_text}, {over_text})"


def _b2b_input(b: dict, size: int, min_share: float) -> str:
    known = b.get("known_role", 0)
    if b.get("b2b_insufficient"):
        return (f"insufficient: {known} of {size} pains have a known role (under "
                f"{min_share:.0%}); global prior {b.get('b2b_prior', 0):.0%} used")
    return (f"{b.get('business_count', 0)} of {known} pains with a known role are businesses "
            f"({known} of {size} known)")


def _automated(b: dict) -> list[str]:
    auto = b.get("auto") or {}
    if not auto:
        return ["_No automated review yet: run `niche-miner review`._", ""]
    comp = auto.get("competition") or {}
    when = str(auto.get("fetched_at") or "")[:10]
    lines = []
    if comp:
        incumbents = " · ".join(f"{i['name']} ({i.get('country', '?')}, {i.get('fit', '?')})"
                                for i in comp.get("incumbents") or []) or "none found"
        lines += [f"**Incumbents** (automated {when}, competition {comp.get('score')}/10): "
                  f"{incumbents}", "", f"_Why:_ {comp.get('rationale', '')}", ""]
    return lines


def _who_line(b: dict, unknown_weight: float) -> list[str]:
    counts = b.get("who_counts") or {}
    if not b.get("buyer_roles_set"):
        return ["**who_match:** 1.00 — no `buyer_roles` set for this niche in niches.yaml, so "
                "no discount.", ""]
    return [f"**who_match:** {b.get('who_match', 0):.2f} — {counts.get('buyer', 0)} pains from "
            f"buyer roles, {counts.get('unknown', 0)} without `who` (×{unknown_weight:g}), "
            f"{counts.get('other', 0)} from other roles (×0).", ""]


def _cluster_section(conn: sqlite3.Connection, c: sqlite3.Row, absorbed: list[int],
                     rk: tables.Ranks, settings: Settings) -> list[str]:
    members = db.member_details(conn, [c["id"], *absorbed])
    scored = [m for m in members if not m["exclude_from_scoring"]]
    b = bd(c)
    size = c["size"] or 1
    by_source = ", ".join(f"{s} {n}" for s, n in Counter(m["source"] for m in scored).most_common())
    meta = (f"cluster `{c['id']}` · niche `{c['niche_tag']}` · {c['size']} pains · "
            f"{c['distinct_authors']} distinct authors · {b.get('distinct_urls', 0)} source "
            f"documents · {b.get('distinct_sources', 0)} source(s) ({by_source})")
    if excluded := len(members) - len(scored):
        meta += f" · {excluded} more from outside Colombia, not scored"
    if absorbed:
        meta += f" · merged from {', '.join(map(str, absorbed))}"
    rank, co_rank = rk[c["id"]]
    reviewed = "reviewed" if b.get("reviewed") else "not reviewed"
    status = (f"Status: `{b.get('status', 'active')}` ({reviewed}) — to drop it, add "
              f"`  {c['id']}: {{status: not_a_niche}}` under `clusters:` in "
              "`config/reachability.yaml` and re-run `score` and `report`.")
    lines = [f"### {rank}. {c['label']} — {c['score']:.1f}", "", meta, "", status, ""]
    if flagged := tables.flags(c, rk, settings.report):
        lines += ["**Flags:** " + " · ".join(flagged), ""]
    if b.get("top_url_share", 0) > settings.report.single_url_flag:
        lines += [f"⚠ **One document holds {b['top_url_count']} of {size} pains "
                  f"({b['top_url_share']:.0%}):** {b['top_url']}", ""]
    types = " · ".join(f"{t} {n}" for t, n in Counter(m["pain_type"] or "unknown"
                                                       for m in scored).most_common())
    co = b.get("co_only")
    co_note = (f"Colombia only {co['total']:.1f} from {co['size']} pains (rank {co_rank} vs "
               f"{rank})" if co else "no pains with country co")
    lines += ["| Component | Input | Points |", "|---|---|---:|",
              f"| Frequency | {b.get('distinct_urls', 0)} documents, {c['distinct_authors']} "
              f"authors; Σ ln(1 + authors per document) = {b.get('url_spread', 0):.2f} "
              f"| +{b.get('frequency', 0):.1f} (old formula +{b.get('frequency_old', 0):.1f}) |",
              f"| Spend | {c['spend_count']} of {size} pains with a spend signal "
              f"| +{b.get('spend', 0):.1f} |",
              f"| B2B | {_b2b_input(b, size, settings.scoring.b2b_min_known_share)} "
              f"| +{b.get('b2b', 0):.1f} |",
              f"| Pain-type weight | {types} | ×{b.get('type_weight', 1):.2f} |",
              f"| Reachability | {_input(b, 'reachability')} | +{b.get('reachability', 0):.1f} |",
              f"| Competition | {_input(b, 'competition')} | −{b.get('competition', 0):.1f} |",
              f"| **Score** | evidence {b.get('evidence', 0):.1f}; with the old frequency "
              f"{b.get('total_old_frequency', 0):.1f}; {co_note} | **{c['score']:.1f}** |", ""]
    lines += _automated(b) + _who_line(b, settings.scoring.unknown_who_weight)
    lines += ["**Representative quotes**", ""]
    for i, m in enumerate(_representatives(c, scored), 1):
        link = f"[{m['source']}]({m['url']})" if m["url"] else m["source"]
        lines += [f"{i}. “{_quote(m['text'])}” — {link}", f"   - _Dolor:_ {m['pain']}"]
    lines += ["", f"**Who:** {_distribution(m['who'] for m in scored)}", "",
              f"**Current tool:** {_distribution(m['current_tool'] for m in scored)}", ""]
    for field, title in (("spend_signal", "Spend signals"), ("cost_of_problem",
                                                             "Cost of the problem (not spend)")):
        quotes = [m[field] for m in scored if m[field]]
        if quotes:
            lines += [f"**{title}:** " + " · ".join(f"“{_quote(q)}”" for q in quotes[:3]), ""]
    return lines + ["**Hypothesis:** ", "", "---", ""]
