"""Report tables: the top-clusters summary, every cluster in the latest run, the clusters not
yet reviewed, and a paste-ready overrides stub for config/reachability.yaml."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

from .settings import ReportSettings

Ranks = dict[int, tuple[int, int | None]]  # cluster id -> (rank, co-only rank)


def bd(c: sqlite3.Row) -> dict:
    return json.loads(c["score_breakdown"]) if c["score_breakdown"] else {}


def cell(text: str) -> str:
    return text.replace("|", "\\|")


def rank_key(c: sqlite3.Row) -> tuple[float, float]:
    return c["score"] or 0.0, bd(c).get("evidence", 0.0)


def co_total(c: sqlite3.Row) -> float | None:
    co = bd(c).get("co_only")
    return co["total"] if co else None


def ranks(active: Sequence[sqlite3.Row]) -> Ranks:
    """Rank by score and by the co-only score; clusters with no co pains have no co rank."""
    with_co = sorted((c for c in active if co_total(c) is not None),
                     key=lambda c: (co_total(c), bd(c).get("evidence", 0.0)), reverse=True)
    co_rank = {c["id"]: i for i, c in enumerate(with_co, 1)}
    return {c["id"]: (i, co_rank.get(c["id"])) for i, c in enumerate(active, 1)}


def flags(c: sqlite3.Row, rk: Ranks, cfg: ReportSettings) -> list[str]:
    b = bd(c)
    out = []
    if b.get("no_audience"):
        out.append("🔴 no visible audience")
    if b.get("top_url_share", 0) > cfg.single_url_flag:
        out.append(f"⚠ one document {b['top_url_share']:.0%}")
    rank, co = rk.get(c["id"], (0, None))
    if co is None and rank:
        out.append("no co pains")
    elif rank and abs(co - rank) > cfg.rank_change_flag:
        out.append(f"co-only rank {rank}→{co}")
    if b.get("b2b_insufficient"):
        out.append("b2b insufficient")
    return out


def value_cell(b: dict, part: str) -> str:
    value, source = b.get(f"{part}_value"), b.get(f"{part}_from")
    if value is None:
        return "—"
    return f"{value:g} ({'override' if source == 'override' else 'auto'})"


def _num(x: float | None) -> str:
    return f"{x:.1f}" if x is not None else "—"


def top_table(top: Sequence[sqlite3.Row], rk: Ranks, cfg: ReportSettings) -> list[str]:
    lines = ["| # | id | theme | score | old freq. | co-only | docs · sources | top doc "
             "| reach | comp. | flags |",
             "|---:|---:|---|---:|---:|---:|---|---:|---|---|---|"]
    for c in top:
        b = bd(c)
        lines.append(
            f"| {rk[c['id']][0]} | {c['id']} | {cell(c['label'])} | **{c['score']:.1f}** | "
            f"{_num(b.get('total_old_frequency'))} | {_num(co_total(c))} | "
            f"{b.get('distinct_urls', 0)} · {b.get('distinct_sources', 0)} | "
            f"{b.get('top_url_share', 0):.0%} | {value_cell(b, 'reachability')} | "
            f"{value_cell(b, 'competition')} | {', '.join(flags(c, rk, cfg))} |")
    return lines + ["", "_score_ is the current formula; _old freq._ is the same score with the "
                    "old frequency (distinct authors, not documents); _co-only_ uses only pains "
                    "with country `co`. _top doc_ is the share of pains from the busiest "
                    "document (video, app listing, place, post).", ""]


def all_clusters(ranked: Sequence[sqlite3.Row], clusters: Sequence[sqlite3.Row],
                 rk: Ranks) -> list[str]:
    if not clusters:
        return []
    lines = ["## All clusters in the latest run", "",
             "| id | label | status | pains | docs | sources | top doc | authors | spend | b2b "
             "| reach | comp. | score | old freq. | co-only |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|"]
    for c in ranked:
        b = bd(c)
        b2b = "insufficient" if b.get("b2b_insufficient") else f"{c['b2b_ratio']:.0%}"
        lines.append(
            f"| {c['id']} | {cell(c['label'])} | {b.get('status', 'active')} | {c['size']} | "
            f"{b.get('distinct_urls', 0)} | {b.get('distinct_sources', 0)} | "
            f"{b.get('top_url_share', 0):.0%} | {c['distinct_authors']} | {c['spend_count']} | "
            f"{b2b} | {value_cell(b, 'reachability')} | {value_cell(b, 'competition')} | "
            f"{c['score']:.1f} | {_num(b.get('total_old_frequency'))} | {_num(co_total(c))} |")
    for c in clusters:
        if c["merged_into"]:
            note = f"merged into {c['merged_into']}"
        elif c["is_noise"]:
            note = "noise, not scored"
        elif c["score"] is None:
            note = "not scored (no pains from Colombia)"
        else:
            continue
        lines.append(f"| {c['id']} | _{cell(c['label'])}_ | {note} | {c['size']} | | | | "
                     f"{c['distinct_authors']} | {c['spend_count']} | | | | | | |")
    return lines + [""]


def _auto_note(b: dict) -> str:
    auto = [f"{k} {b[f'{k}_auto']:g}" for k in ("reachability", "competition")
            if b.get(f"{k}_auto") is not None]
    return f"automated: {', '.join(auto)}" if auto else "no automated review yet"


def not_reviewed(ranked: Sequence[sqlite3.Row]) -> list[str]:
    pending = [c for c in ranked if not bd(c).get("reviewed")]
    lines = ["## Not yet reviewed", ""]
    if not pending:
        return lines + ["Every scored cluster has an override, a status or a note.", ""]
    lines += [f"{len(pending)} of {len(ranked)} scored clusters have no override, status or note "
              "in `config/reachability.yaml` (their scores use the automated values):", ""]
    lines += [f"- `{c['id']}` {c['label']} ({_auto_note(bd(c))})" for c in pending]
    return lines + [""]


def overrides_stub(ranked: Sequence[sqlite3.Row]) -> list[str]:
    pending = [c for c in ranked if not bd(c).get("reviewed")]
    if not pending:
        return []
    lines = ["## Review", "",
             "Paste into `config/reachability.yaml` under `clusters:` and change what you "
             "disagree with: a number overrides the automated value (null keeps it), `status` "
             "is `active`, `not_a_niche` or `merged_into:<id>`, and a note alone marks the "
             "cluster as reviewed. Then run `niche-miner score && niche-miner report`.", "",
             "```yaml", "clusters:"]
    for c in pending:
        lines.append(f"  {c['id']}: {{reachability: null, competition: null, status: active, "
                     f"note: \"\"}}  # {c['label']} ({_auto_note(bd(c))})")
    return lines + ["```", ""]
