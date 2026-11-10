"""Cluster scoring, 0–100. Formula and weights are printed in every report (config: `scoring`).

Reachability and competition come from a human override in config/reachability.yaml when one
is set, else from the automated review (`niche-miner review`): competition per cluster, and
reachability = the niche's reachability_proxy × the cluster's who_match (audience.py).
Pains from a known non-Colombian country stay in their cluster but are left out of the score;
`co_only` repeats the score with only the pains whose country is "co"."""

from __future__ import annotations

import logging
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import db, reviews
from .audience import who_match
from .settings import ScoreWeights, load_niches
from .source_rules import source_key

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusterStats:
    size: int  # scored pains (known non-Colombian ones excluded)
    distinct_authors: int
    spend_count: int
    business_count: int
    type_weight: float = 1.0  # mean pain-type weight of the scored pains
    excluded: int = 0  # members left out of the score (non-Colombian)
    known_role: int = 0  # pains whose writer is known to be a business or a consumer
    distinct_urls: int = 0  # source documents: videos, app listings, places, posts
    distinct_sources: int = 0
    url_spread: float = 0.0  # Σ_document min(ln(1 + authors there), ln(1 + doc cap))
    top_url: str | None = None  # the document with the most pains
    top_url_count: int = 0
    cap_hits: int = 0  # documents with at least frequency_doc_cap authors

    @property
    def top_url_share(self) -> float:
        return self.top_url_count / self.size if self.size else 0.0


@dataclass(frozen=True)
class Breakdown:
    frequency: float
    frequency_old: float  # author-count formula, ln(1 + distinct authors), for comparison
    spend: float
    b2b: float
    b2b_share: float  # business share used: among known roles, or the global prior
    b2b_known: float  # share of pains with a known role
    b2b_insufficient: bool  # too few known roles: the global prior was used
    type_weight: float
    evidence: float  # (frequency + spend + b2b) × type_weight
    reachability: float  # points added
    competition: float  # points subtracted
    total: float
    total_old_frequency: float  # the same score with the old frequency formula
    freq_sum: float = 0.0  # the capped per-document sum (= stats.url_spread)
    cap_hits: int = 0
    above_ceiling: bool = False  # freq_sum ≥ frequency_ceiling: full frequency points


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _role(m: Mapping[str, Any]) -> str:
    """The extraction's who_type, falling back to the source's author_type."""
    who = m["who_type"]
    return who if who not in (None, "unknown") else (m["author_type"] or "unknown")


def compute_stats(members: Sequence[Mapping[str, Any]], w: ScoreWeights) -> ClusterStats:
    """Aggregate member rows (db.member_details) into the numbers the formula uses."""
    scored = [m for m in members if not m["exclude_from_scoring"]]
    author = {id(m): m["author_hash"] or f"signal:{m['signal_id']}" for m in scored}
    by_url: dict[str, set[str]] = defaultdict(set)
    pains_by_url: Counter[str] = Counter()
    for m in scored:
        key = source_key(m["source"], m["url"], m["signal_id"])
        by_url[key].add(author[id(m)])
        pains_by_url[key] += 1
    roles = [_role(m) for m in scored]
    weights = [w.pain_type_weights.get(m["pain_type"] or "", 1.0) for m in scored]
    top = pains_by_url.most_common(1)
    cap = math.log1p(w.frequency_doc_cap)
    return ClusterStats(
        size=len(scored), distinct_authors=len(set(author.values())),
        spend_count=sum(1 for m in scored if m["spend_signal"]),
        business_count=roles.count("business"),
        type_weight=sum(weights) / len(weights) if weights else 1.0,
        excluded=len(members) - len(scored),
        known_role=sum(1 for r in roles if r in ("business", "consumer")),
        distinct_urls=len(by_url), distinct_sources=len({m["source"] for m in scored}),
        url_spread=sum(min(math.log1p(len(a)), cap) for a in by_url.values()),
        top_url=top[0][0] if top else None, top_url_count=top[0][1] if top else 0,
        cap_hits=sum(1 for a in by_url.values() if len(a) >= w.frequency_doc_cap))


def compute_score(stats: ClusterStats, reachability: float, competition: float,
                  w: ScoreWeights, b2b_prior: float = 0.0) -> Breakdown:
    n = max(stats.size, 1)
    frequency = w.frequency * min(1.0, stats.url_spread / w.frequency_ceiling)
    frequency_old = w.frequency * min(1.0, math.log1p(stats.distinct_authors)
                                      / math.log1p(w.frequency_saturation))
    spend = w.spend * stats.spend_count / n
    known = stats.known_role / n
    off = w.b2b == 0  # weight 0: the component is off, so no prior and no flag
    insufficient = not off and (stats.size == 0 or known < w.b2b_min_known_share)
    share = 0.0 if off else b2b_prior if insufficient else stats.business_count / stats.known_role
    b2b = w.b2b * share
    reach_pts = w.reachability * _clamp(reachability, 0, 10) / 10
    comp_pts = w.competition * _clamp(competition, 0, 10) / 10

    def total(freq: float) -> float:
        return _clamp((freq + spend + b2b) * stats.type_weight + reach_pts - comp_pts, 0, 100)

    r = lambda x: round(x, 1)  # noqa: E731
    return Breakdown(r(frequency), r(frequency_old), r(spend), r(b2b), round(share, 3),
                     round(known, 3), insufficient, round(stats.type_weight, 2),
                     r((frequency + spend + b2b) * stats.type_weight), r(reach_pts), r(comp_pts),
                     r(total(frequency)), r(total(frequency_old)), round(stats.url_spread, 2),
                     stats.cap_hits, stats.url_spread >= w.frequency_ceiling)


def formula_markdown(w: ScoreWeights) -> str:
    weights = ", ".join(f"{k} {v:g}" for k, v in sorted(w.pain_type_weights.items()))
    return "\n".join([
        "```",
        "score = clamp(evidence + reachability - competition, 0, 100)",
        "  evidence     = (frequency + spend + b2b) × type_weight",
        f"  frequency    = {w.frequency:g} × min(1, Σ_document min(ln(1 + authors_document), "
        f"ln(1 + {w.frequency_doc_cap})) / {w.frequency_ceiling:g})",
        f"                 (ceiling {w.frequency_ceiling:g}: " + (
            f"calibrated {w.frequency_recalibrated_at}" if w.frequency_recalibrated_at
            else "set by hand, `niche-miner calibrate` not run yet") + ")",
        f"                 (old, for comparison: {w.frequency:g} × min(1, ln(1 + distinct_authors)"
        f" / ln(1 + {w.frequency_saturation})))",
        f"  spend        = {w.spend:g} × pains_with_spend_signal / pains",
        *(["  b2b          = 0 (off: weight 0 in settings)"] if w.b2b == 0 else [
            f"  b2b          = {w.b2b:g} × business_pains / pains_with_known_role",
            f"                 (global prior instead when under {w.b2b_min_known_share:.0%} of "
            "pains have a known role)"]),
        f"  type_weight  = mean pain-type weight of the cluster's pains ({weights or 'none'}; "
        "other types 1)",
        f"  reachability = {w.reachability:g} × reachability / 10    "
        "(override, else niche reachability_proxy × who_match, 0-10)",
        f"                 who_match = (buyer pains + {w.unknown_who_weight:g} × pains without "
        "`who`) / pains",
        f"  competition  = {w.competition:g} × competition / 10     "
        "(override, else automated competition, 0-10, subtracted)",
        f"  flag 'no visible audience' when the automated reachability < "
        f"{w.no_audience_below:g} and there is no override (still ranked)",
        "```",
        "A document is one video, app listing, Maps place, TikTok video or job post. Weights: "
        "`config/settings.yaml` → `scoring`. Pains from a known non-Colombian country are kept "
        "but not scored; *co-only* repeats the score with only pains whose country is `co`. "
        "The noise cluster is never scored.",
    ])


def _b2b_priors(conn: sqlite3.Connection, clusters: Sequence[sqlite3.Row],
                w: ScoreWeights) -> dict[str, float]:
    """Business share among pains with a known role, per niche, over the whole latest run."""
    ids: dict[str, list[int]] = defaultdict(list)
    for c in clusters:
        ids[c["niche_tag"]].append(c["id"])
    priors = {}
    for niche, cids in ids.items():
        s = compute_stats(db.member_details(conn, cids), w)
        priors[niche] = s.business_count / s.known_role if s.known_role else 0.0
    return priors


def _pick(override: float | None, auto: Any) -> tuple[float | None, str]:
    if override is not None:
        return override, "override"
    if isinstance(auto, int | float) and not isinstance(auto, bool):
        return float(auto), "automated"
    return None, "none"


def _auto_part(auto: Mapping[str, Any], part: str) -> dict[str, Any]:
    value = auto.get(part)
    return value if isinstance(value, dict) else {}


def resolve_groups(conn: sqlite3.Connection, config_dir: Path
                   ) -> tuple[list[sqlite3.Row], dict, dict[int, int], dict[int, list[int]]]:
    """Latest clusters, overrides, and each cluster's owner after merges (merges.yaml plus
    `merged_into:` statuses), with the member clusters of every owner."""
    clusters = db.latest_clusters(conn)
    current = {c["id"] for c in clusters if not c["is_noise"]}
    overrides, status_merges, warnings = reviews.load_overrides(
        config_dir / "reachability.yaml", current)
    merges = reviews.load_merges(config_dir / "merges.yaml")
    for target, sources in status_merges.items():
        merges.setdefault(target, []).extend(sources)
    owner, more = reviews.resolve_merges(current, merges)
    for message in warnings + more:
        log.warning(message)
    groups: dict[int, list[int]] = defaultdict(list)
    for cid, target in owner.items():
        groups[target].append(cid)
    return clusters, overrides, owner, groups


def score_latest(conn: sqlite3.Connection, config_dir: Path, w: ScoreWeights,
                 review_dir: Path | None = None) -> list[dict]:
    """Recompute metrics and scores for the latest run of every niche. Idempotent."""
    clusters, overrides, owner, groups = resolve_groups(conn, config_dir)
    priors = _b2b_priors(conn, clusters, w)
    niches = load_niches(config_dir)
    niche_auto = {c["niche_tag"]: reviews.load_niche_auto(review_dir, c["niche_tag"])
                  for c in clusters}
    scored: list[dict] = []
    with conn:
        for c in clusters:
            cid = c["id"]
            if c["is_noise"] or owner[cid] != cid:
                stats = compute_stats(db.member_details(conn, [cid]), w)
                db.save_score(conn, cid, stats, merged_into=None if c["is_noise"] else owner[cid])
                continue
            members = db.member_details(conn, sorted(groups[cid]))
            stats = compute_stats(members, w)
            if stats.size == 0:  # every member is from outside Colombia
                db.save_score(conn, cid, stats)
                continue
            o = overrides.get(cid, reviews.Override())
            auto = reviews.load_auto(review_dir, cid)
            comp_auto = _auto_part(auto, "competition")
            proxy = niche_auto[c["niche_tag"]].get("score")
            roles = [str(r) for r in niches.get(c["niche_tag"], {}).get("buyer_roles") or []]
            match, who_counts = who_match([m for m in members if not m["exclude_from_scoring"]],
                                          roles, w.unknown_who_weight)
            reach_auto = (round(proxy * match, 1) if isinstance(proxy, int | float)
                          and not isinstance(proxy, bool) else None)
            reach, reach_from = _pick(o.reachability, reach_auto)
            comp, comp_from = _pick(o.competition, comp_auto.get("score"))
            prior = priors.get(c["niche_tag"], 0.0)
            b = compute_score(stats, reach or 0, comp or 0, w, prior)
            co = compute_stats([m for m in members if m["country"] == "co"], w)
            co_b = compute_score(co, reach or 0, comp or 0, w, prior) if co.size else None
            data = asdict(b) | {
                "status": o.status, "reviewed": o.reviewed, "note": o.note,
                "reachability_value": reach, "reachability_from": reach_from,
                "reachability_auto": reach_auto, "reachability_override": o.reachability,
                "niche_proxy": proxy, "who_match": round(match, 3), "who_counts": who_counts,
                "buyer_roles_set": bool(roles),
                "competition_value": comp, "competition_from": comp_from,
                "competition_auto": comp_auto.get("score"), "competition_override": o.competition,
                "no_audience": o.reachability is None and (reach_auto is None
                                                           or reach_auto < w.no_audience_below),
                "distinct_urls": stats.distinct_urls, "distinct_sources": stats.distinct_sources,
                "url_spread": round(stats.url_spread, 2), "top_url": stats.top_url,
                "frequency_ceiling": w.frequency_ceiling,
                "frequency_doc_cap": w.frequency_doc_cap,
                "top_url_count": stats.top_url_count, "top_url_share": round(stats.top_url_share, 3),
                "known_role": stats.known_role, "business_count": stats.business_count,
                "b2b_prior": round(prior, 3),
                "co_only": {"total": co_b.total, "size": co.size} if co_b else None,
                "auto": {"fetched_at": auto.get("fetched_at"), "competition": comp_auto}
                if auto else None}
            db.save_score(conn, cid, stats, reachability=reach or 0, competition=comp or 0,
                          score=b.total, breakdown=data)
            scored.append({"id": cid, "label": c["label"], "niche": c["niche_tag"],
                           "score": b.total, "old": b.total_old_frequency,
                           "co_only": co_b.total if co_b else None, "status": o.status})
    scored.sort(key=lambda s: s["score"], reverse=True)
    return scored
