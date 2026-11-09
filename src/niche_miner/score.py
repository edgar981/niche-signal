"""Cluster scoring, 0–100. Formula and weights are printed in every report (config: `scoring`).

Manual inputs, keyed by global cluster id from the latest report:
  config/reachability.yaml  reachability / competition (0–10) per cluster
  config/merges.yaml        target_id: [ids folded into it]
Pains from a known non-Colombian country stay in their cluster but are left out of the score."""

from __future__ import annotations

import logging
import math
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import db
from .settings import ScoreWeights, load_yaml

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusterStats:
    size: int  # scored pains (known non-Colombian ones excluded)
    distinct_authors: int
    spend_count: int
    business_count: int
    type_weight: float = 1.0  # mean pain-type weight of the scored pains
    excluded: int = 0  # members left out of the score (non-Colombian)


@dataclass(frozen=True)
class Breakdown:
    frequency: float
    spend: float
    b2b: float
    type_weight: float
    evidence: float  # (frequency + spend + b2b) × type_weight
    reachability: float
    competition: float  # points subtracted
    uncapped: float
    capped: bool
    reviewed: bool  # cluster has an entry in reachability.yaml
    total: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_stats(members: Sequence[Mapping[str, Any]], w: ScoreWeights) -> ClusterStats:
    """Aggregate member rows (db.member_details) into the numbers the formula uses.
    Business = the extraction's who_type, falling back to the source's author_type."""
    scored = [m for m in members if not m["exclude_from_scoring"]]
    authors = {m["author_hash"] or f"signal:{m['signal_id']}" for m in scored}
    business = sum(
        1 for m in scored
        if (m["who_type"] if m["who_type"] not in (None, "unknown") else m["author_type"])
        == "business")
    weights = [w.pain_type_weights.get(m["pain_type"] or "", 1.0) for m in scored]
    return ClusterStats(
        size=len(scored), distinct_authors=len(authors),
        spend_count=sum(1 for m in scored if m["spend_signal"]), business_count=business,
        type_weight=sum(weights) / len(weights) if weights else 1.0,
        excluded=len(members) - len(scored))


def compute_score(stats: ClusterStats, reachability: float, competition: float,
                  w: ScoreWeights, reviewed: bool = False) -> Breakdown:
    n = max(stats.size, 1)
    reach = _clamp(reachability, 0, 10)
    comp = _clamp(competition, 0, 10)
    frequency = w.frequency * min(
        1.0, math.log1p(stats.distinct_authors) / math.log1p(w.frequency_saturation))
    spend = w.spend * stats.spend_count / n
    b2b = w.b2b * stats.business_count / n
    evidence = (frequency + spend + b2b) * stats.type_weight
    reach_pts = w.reachability * reach / 10
    comp_pts = w.competition * comp / 10
    uncapped = _clamp(evidence + reach_pts - comp_pts, 0, 100)
    capped = reach == 0 and uncapped > w.unreachable_cap
    total = min(uncapped, w.unreachable_cap) if reach == 0 else uncapped
    r = lambda x: round(x, 1)  # noqa: E731
    return Breakdown(r(frequency), r(spend), r(b2b), round(stats.type_weight, 2), r(evidence),
                     r(reach_pts), r(comp_pts), r(uncapped), capped, reviewed, r(total))


def formula_markdown(w: ScoreWeights) -> str:
    weights = ", ".join(f"{k} {v:g}" for k, v in sorted(w.pain_type_weights.items()))
    return "\n".join([
        "```",
        "score = clamp(evidence + reachability - competition, 0, 100)",
        "  evidence     = (frequency + spend + b2b) × type_weight",
        f"  frequency    = {w.frequency:g} × min(1, ln(1 + distinct_authors) / "
        f"ln(1 + {w.frequency_saturation}))",
        f"  spend        = {w.spend:g} × pains_with_spend_signal / pains",
        f"  b2b          = {w.b2b:g} × business_pains / pains    "
        "(who_type from extraction, else the source's author_type)",
        f"  type_weight  = mean pain-type weight of the cluster's pains ({weights or 'none'}; "
        "other types 1)",
        f"  reachability = {w.reachability:g} × reachability / 10    "
        "(config/reachability.yaml, 0-10)",
        f"  competition  = {w.competition:g} × competition / 10     "
        "(config/reachability.yaml, 0-10, subtracted)",
        f"  if reachability = 0: score = min(score, {w.unreachable_cap:g})",
        "```",
        "Weights: `config/settings.yaml` → `scoring`. Pains from a known non-Colombian country "
        "are kept but not scored, and the noise cluster is never scored.",
    ])


def load_merges(path: Path) -> dict[int, list[int]]:
    raw = load_yaml(path).get("merges") or {}
    return {int(target): [int(s) for s in (sources or [])] for target, sources in raw.items()}


def load_reviews(path: Path) -> dict[int, dict[str, Any]]:
    raw = load_yaml(path).get("clusters") or {}
    return {int(cid): (entry or {}) for cid, entry in raw.items()}


def resolve_merges(cluster_ids: set[int],
                   merges: dict[int, list[int]]) -> tuple[dict[int, int], list[str]]:
    """Map every current cluster id to the id of the group it ends up in (chains allowed).
    Ids that aren't in the latest run are reported and ignored."""
    owner = {cid: cid for cid in cluster_ids}
    warnings: list[str] = []
    for target, sources in merges.items():
        if target not in owner:
            warnings.append(f"merges.yaml: target {target} is not in the latest run; skipped")
            continue
        for source in sources:
            if source not in owner:
                warnings.append(f"merges.yaml: {source} is not in the latest run; skipped")
                continue
            old, new = owner[source], owner[target]
            if old != new:
                for cid, own in owner.items():
                    if own == old:
                        owner[cid] = new
    return owner, warnings


def score_latest(conn: sqlite3.Connection, config_dir: Path, w: ScoreWeights) -> list[dict]:
    """Recompute metrics and scores for the latest run of every niche. Idempotent."""
    clusters = db.latest_clusters(conn)
    current = {c["id"] for c in clusters if not c["is_noise"]}
    owner, warnings = resolve_merges(current, load_merges(config_dir / "merges.yaml"))
    for message in warnings:
        log.warning(message)
    reviews = load_reviews(config_dir / "reachability.yaml")
    groups: dict[int, list[int]] = defaultdict(list)
    for cid, target in owner.items():
        groups[target].append(cid)
    scored: list[dict] = []
    with conn:
        for c in clusters:
            cid = c["id"]
            if c["is_noise"] or owner[cid] != cid:
                stats = compute_stats(db.member_details(conn, [cid]), w)
                db.save_score(conn, cid, stats, merged_into=None if c["is_noise"] else owner[cid])
                if not c["is_noise"] and cid in reviews:
                    log.warning("reachability.yaml: %d was merged into %d; score the target",
                                cid, owner[cid])
                continue
            stats = compute_stats(db.member_details(conn, sorted(groups[cid])), w)
            if stats.size == 0:  # every member is from outside Colombia
                db.save_score(conn, cid, stats)
                continue
            review = reviews.get(cid, {})
            reach = float(review.get("reachability") or 0)
            comp = float(review.get("competition") or 0)
            b = compute_score(stats, reach, comp, w, reviewed=cid in reviews)
            db.save_score(conn, cid, stats, reachability=reach, competition=comp, score=b.total,
                          breakdown=asdict(b))
            scored.append({"id": cid, "label": c["label"], "niche": c["niche_tag"],
                           "score": b.total, "uncapped": b.uncapped})
    scored.sort(key=lambda s: (s["score"], s["uncapped"]), reverse=True)
    return scored
