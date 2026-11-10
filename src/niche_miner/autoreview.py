"""`niche-miner review`: the automated inputs that `score` uses when there is no override.

  competition   per active cluster of the latest run (not not_a_niche, not merged away),
                cached in data/review/<cluster_id>.json
  reachability  per niche, by channel type (audience.py), cached in data/review/niche_<tag>.json;
                `score` turns it into a cluster value with who_match

Both are reused for `review.cache_days`. What it spends: one Apify Google-search run for
everything that needs it (about $0.0045 per query: 3 per cluster, 1 per niche), one Claude call
per cluster and per niche, a few Computrabajo pages (3 s apart) and YouTube API units. Without
APIFY_TOKEN, YOUTUBE_API_KEY or a configured search actor, that input is skipped and logged.
Cluster ids are global, so a new `cluster` run needs a new competition review."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from apify_client.errors import ApifyApiError

from . import db, reviews
from .audience import (creator_points, find_gremios, niche_score, parse_offer_count,
                       tiktok_accounts, youtube_video_ids)
from .collectors.apify_common import ENV_KEY, render, run_summary, start_and_wait
from .collectors.base import StopSource
from .collectors.computrabajo import ComputrabajoCollector, search_url
from .collectors.robots import Robots
from .collectors.youtube import YouTubeCollector
from .competition import assess, build_queries, current_tools, parse_search, play_search
from .extract import Completer
from .llm import LLMError
from .settings import Settings

log = logging.getLogger(__name__)


@dataclass
class ReviewStats:
    reviewed: int = 0  # clusters
    cached: int = 0
    failed: int = 0
    niches_reviewed: int = 0
    niches_cached: int = 0
    search_usd: float = 0.0


def active_groups(conn: sqlite3.Connection, config_dir: Path,
                  niche: str | None = None) -> list[tuple[sqlite3.Row, list[int]]]:
    """(cluster, member cluster ids) for every cluster that is scored on its own."""
    clusters = [c for c in db.latest_clusters(conn, niche) if not c["is_noise"]]
    ids = {c["id"] for c in clusters}
    overrides, status_merges, _ = reviews.load_overrides(config_dir / "reachability.yaml", ids)
    merges = reviews.load_merges(config_dir / "merges.yaml")
    for target, sources in status_merges.items():
        merges.setdefault(target, []).extend(sources)
    owner, _ = reviews.resolve_merges(ids, merges)
    return [(c, sorted(i for i, o in owner.items() if o == c["id"])) for c in clusters
            if owner[c["id"]] == c["id"]
            and overrides.get(c["id"], reviews.Override()).status != "not_a_niche"]


def is_fresh(data: dict[str, Any], days: int, now: datetime, part: str = "competition") -> bool:
    try:
        fetched = datetime.fromisoformat(str(data.get("fetched_at")))
    except ValueError:
        return False
    return bool(data.get(part)) and now - fetched < timedelta(days=days)


class Reviewer:
    def __init__(self, conn: sqlite3.Connection, settings: Settings, config_dir: Path,
                 review_dir: Path, llm: Completer, niches: dict[str, dict[str, Any]],
                 now: datetime | None = None):
        self.conn, self.settings, self.cfg = conn, settings, settings.review
        self.config_dir, self.review_dir, self.llm, self.niches = config_dir, review_dir, llm, niches
        self.model = self.cfg.model or settings.llm.label_model
        self.now = now or datetime.now(UTC)
        self.stats = ReviewStats()
        self.ct = ComputrabajoCollector(settings)
        self.ct_stopped = False
        self.yt = YouTubeCollector(settings)
        self._offers: dict[tuple[str, str], int | None] = {}

    def run(self, niche: str | None = None, refresh: bool = False) -> ReviewStats:
        groups = active_groups(self.conn, self.config_dir, niche)
        stale = []
        for c, ids in groups:
            if not refresh and is_fresh(reviews.load_auto(self.review_dir, c["id"]),
                                        self.cfg.cache_days, self.now):
                self.stats.cached += 1
            else:
                stale.append((c, ids))
        stale_niches = []
        for tag in sorted({c["niche_tag"] for c, _ in groups}):
            cached = reviews.load_niche_auto(self.review_dir, tag)
            searched = ((cached.get("types") or {}).get("gremio") or {}).get("searched")
            if not refresh and searched and is_fresh(cached, self.cfg.cache_days, self.now,
                                                     part="types"):
                self.stats.niches_cached += 1
            else:
                stale_niches.append(tag)
        queries = {c["id"]: build_queries(c["label"], self.cfg.query_suffixes) for c, _ in stale}
        gremio_q = {t: self.cfg.gremio_query.format(niche=self.niche_name(t)) for t in stale_niches}
        search = self._google(sorted({q for qs in queries.values() for q in qs}
                                     | set(gremio_q.values())))
        self.review_dir.mkdir(parents=True, exist_ok=True)
        for c, ids in stale:
            members = [m for m in db.member_details(self.conn, ids) if not m["exclude_from_scoring"]]
            self._write(reviews.review_path(self.review_dir, c["id"]),
                        self._competition(c, members, queries[c["id"]], search))
        for tag in stale_niches:
            self._write(reviews.niche_review_path(self.review_dir, tag),
                        self._niche(tag, gremio_q[tag], search))
        return self.stats

    def niche_name(self, tag: str) -> str:
        return str(self.niches.get(tag, {}).get("name") or tag.replace("_", " "))

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def _competition(self, c: sqlite3.Row, members: list[sqlite3.Row], queries: list[str],
                     search: dict[str, list[dict]]) -> dict[str, Any]:
        results = {q: search.get(q, []) for q in queries} if search else {}
        play = play_search(c["label"], self.cfg.play_hits)
        tools = current_tools(members)
        try:
            comp = assess(self.llm, self.model, c["label"],
                          [m["pain"] for m in members[: self.cfg.pains_shown]], results, play, tools)
            self.stats.reviewed += 1
        except LLMError as e:
            log.warning("review: competition for cluster %d failed: %s", c["id"], e)
            comp = None
            self.stats.failed += 1
        log.info("review %d: competition %s — %s", c["id"], comp["score"] if comp else "failed",
                 c["label"])
        return {"cluster_id": c["id"], "label": c["label"], "niche": c["niche_tag"],
                "run_id": c["run_id"], "fetched_at": self.now.isoformat(timespec="seconds"),
                "queries": queries, "search": results, "play": play, "current_tools": tools,
                "competition": comp}

    def _niche(self, tag: str, gremio_query: str, search: dict[str, list[dict]]) -> dict[str, Any]:
        cfg, rc = self.niches.get(tag, {}), self.cfg
        queries = [str(q) for q in cfg.get("computrabajo_queries") or []]
        detail = [{"query": q, "city": city, "posts": self._offer_count(q, city)}
                  for q in queries for city in rc.cities]
        posts = sum(d["posts"] or 0 for d in detail)
        member_map = {str(k): int(v) for k, v in (cfg.get("facebook_group_members") or {}).items()}
        groups = [str(g) for g in cfg.get("facebook_groups") or []]
        groups += [g for g in member_map if g not in groups]
        members = sum(member_map.values()) if member_map else None
        results = search.get(gremio_query, [])
        try:
            gremios = find_gremios(self.llm, self.model, self.niche_name(tag), results)
        except LLMError as e:
            log.warning("review: gremio search for %s failed: %s", tag, e)
            gremios = []
        communities = list(cfg.get("whatsapp_communities") or [])
        types = {
            "job_board": {"signal": posts > 0, "strong": posts >= rc.strong_job_posts,
                          "posts": posts, "queries": queries, "cities": rc.cities,
                          "detail": detail},
            "fb_groups": {"signal": bool(groups), "strong": (members or 0) >= rc.strong_fb_members,
                          "groups": groups, "members": members},
            "gremio": {"signal": bool(gremios),
                       "strong": any(g.get("member_directory") for g in gremios),
                       "query": gremio_query, "searched": bool(search), "results": len(results),
                       "gremios": gremios},
            "whatsapp_community": {"signal": bool(communities), "strong": False,
                                   "communities": communities},
        }
        creator = self._creators(tag)
        score, why = niche_score(types, creator)
        self.stats.niches_reviewed += 1
        log.info("review niche %s: reachability_proxy %d (%s)", tag, score, why)
        return {"niche": tag, "name": self.niche_name(tag),
                "fetched_at": self.now.isoformat(timespec="seconds"), "score": score,
                "rationale": why, "types": types, "creator_comments": creator}

    def _creators(self, tag: str) -> dict[str, Any]:
        rows = self.conn.execute(
            "SELECT source, url FROM raw_signals WHERE niche_tag = ? AND source IN "
            "('youtube', 'apify_tiktok')", (tag,)).fetchall()
        channels: dict[str, dict[str, Any]] = {}
        if self.yt.api_key:
            for vid in youtube_video_ids(rows):
                if (ch := self._fetch_channel(vid)) and ch["id"] not in channels:
                    channels[ch["id"]] = {"name": f"YouTube {ch['title']}",
                                          "audience": ch["subscribers"]}
        handles = tiktok_accounts(rows)
        found = list(channels.values()) + [{"name": f"TikTok @{h}", "audience": None}
                                           for h in handles]
        points = creator_points([c["audience"] for c in channels.values()], handles,
                                self.cfg.strong_creator_audience)
        return {"points": points, "channels": found}

    def _google(self, queries: list[str]) -> dict[str, list[dict]]:
        spec, token = self.cfg.search, os.environ.get(ENV_KEY, "").strip()
        if not queries:
            return {}
        if not token or not spec.actor:
            log.warning("review: no %s or review.search actor; Google results skipped", ENV_KEY)
            return {}
        apify = self.settings.collectors.apify
        try:
            items, done = start_and_wait(token, spec, render(spec.input, {"queries": "\n".join(
                queries)}), len(queries), spec.max_charge_usd or apify.max_charge_usd,
                apify.run_timeout_s)
        except ApifyApiError as e:
            log.warning("review: Google search failed: %s",
                        str(getattr(e, "message", e)).replace(token, "[token]")[:300])
            return {}
        self.stats.search_usd = float(done.usage_total_usd or 0)
        level, status = run_summary(done)
        log.log(level, "review: Google search %s: %d pages for %d queries, about $%.4f so far",
                status, len(items), len(queries), self.stats.search_usd)
        return parse_search(items, self.cfg.results_per_query)

    def _offer_count(self, query: str, city: str) -> int | None:
        """Computrabajo posts in the last 30 days for one job title in one city."""
        key = (query, city)
        if key in self._offers or self.ct_stopped:
            return self._offers.get(key)
        url = f"{search_url(self.ct.cfg.base_url, query)}-en-{city}?pubdate=30"
        try:
            if self.ct.robots is None:
                self.ct.robots = Robots(self.ct.call(self.ct._robots_txt))
            self._offers[key] = parse_offer_count(self.ct.call(self.ct._get, url))
        except StopSource as e:
            log.warning("review: Computrabajo stopped: %s", e)
            self.ct_stopped = True
        except Exception as e:  # SkipTarget, PageGone, network: no count for this one
            log.warning("review: Computrabajo count for %r in %s failed: %s", query, city, e)
            self._offers[key] = None
        return self._offers.get(key)

    def _fetch_channel(self, vid: str) -> dict[str, Any] | None:
        try:
            video = self.yt.call(self.yt._get, "videos", {"part": "snippet", "id": vid})
            snippet = (video.get("items") or [{}])[0].get("snippet") or {}
            if not (cid := snippet.get("channelId")):
                return None
            found = self.yt.call(self.yt._get, "channels", {"part": "statistics", "id": cid})
            stats = (found.get("items") or [{}])[0].get("statistics") or {}
        except Exception as e:  # StopSource, SkipTarget, network
            log.warning("review: YouTube channel for video %s failed: %s", vid, e)
            return None
        subs = None if stats.get("hiddenSubscriberCount") else int(stats.get("subscriberCount") or 0)
        return {"id": cid, "title": snippet.get("channelTitle") or cid, "subscribers": subs}
