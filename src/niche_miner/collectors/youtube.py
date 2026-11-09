"""YouTube comments for the videos listed in config, via the Data API v3 (commentThreads).

Needs YOUTUBE_API_KEY in .env; without it the source is logged and skipped. The key is sent as
a header, never in the URL. Quota: 1 unit for the video's title, 1 per page of 100 threads
(the free default is 10,000 units a day).

Commenters are only hashed (their channel id). Emails, phones, links and @mentions are
scrubbed. Comments by the video's own channel (usually support replies) are left out. The
video title is added as a first line after filtering, so the model knows what a comment is
about without the title counting toward the length or language checks."""

from __future__ import annotations

import html
import json
import logging
import os
import re
from collections.abc import Collection, Iterator
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

from ..models import RawSignal
from ..settings import Settings
from .base import Collector, SkipTarget, StopSource, hash_author
from .pii import scrub

log = logging.getLogger(__name__)

SOURCE = "youtube"
ENV_KEY = "YOUTUBE_API_KEY"
USER_AGENT = "niche-miner/0.1 (personal research)"
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
# Errors that will fail the same way for every video: stop the source instead of skipping.
_STOP_REASONS = {"quotaExceeded", "dailyLimitExceeded", "accessNotConfigured", "keyInvalid",
                 "ipRefererBlocked", "API_KEY_INVALID", "SERVICE_DISABLED",
                 "API_KEY_SERVICE_BLOCKED"}


def video_id(target: str) -> str:
    """An 11-character id, or the id inside a watch / youtu.be / shorts URL."""
    target = target.strip()
    if _VIDEO_ID.fullmatch(target):
        return target
    parts = urlsplit(target)
    candidates = parse_qs(parts.query).get("v", []) + parts.path.strip("/").split("/")[-1:]
    for candidate in candidates:
        if _VIDEO_ID.fullmatch(candidate):
            return candidate
    raise SkipTarget(f"not a YouTube video id or URL: {target!r}")


def comment_url(vid: str, comment_id: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}&lc={comment_id}"


def _date(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value)).astimezone(UTC)
    except ValueError:
        return None


def parse_comment(comment: dict[str, Any], vid: str, niche_tag: str,
                  owner_channel: str | None) -> RawSignal | None:
    """One comment resource (top-level or reply) -> RawSignal; None for the video's own
    channel or an empty text. The display name is never read."""
    snippet = comment.get("snippet") or {}
    comment_id = comment.get("id")
    text = html.unescape(str(snippet.get("textDisplay") or "")).strip()
    author = (snippet.get("authorChannelId") or {}).get("value")
    if not comment_id or not text or (owner_channel and author == owner_channel):
        return None
    return RawSignal(
        niche_tag=niche_tag,
        source=SOURCE,
        external_id=comment_id,
        url=comment_url(vid, comment_id),
        author_hash=hash_author(SOURCE, f"channel:{author}" if author else f"comment:{comment_id}"),
        author_type="unknown",
        text=scrub(text),
        posted_at=_date(snippet.get("publishedAt")),
    )


def parse_threads(payload: dict[str, Any], vid: str, niche_tag: str, owner_channel: str | None,
                  include_replies: bool = True) -> Iterator[RawSignal]:
    for thread in payload.get("items") or []:
        comments = [(thread.get("snippet") or {}).get("topLevelComment") or {}]
        if include_replies:
            comments += (thread.get("replies") or {}).get("comments") or []
        for comment in comments:
            if signal := parse_comment(comment, vid, niche_tag, owner_channel):
                yield signal


def _error(e: HTTPError) -> tuple[set[str], str]:
    """Google's error reasons (e.g. commentsDisabled, quotaExceeded) and message."""
    try:
        body = json.loads(e.read().decode("utf-8"))["error"]
    except (ValueError, KeyError, TypeError, AttributeError):
        return set(), ""
    entries = (body.get("errors") or []) + (body.get("details") or [])
    reasons = {x.get("reason") for x in entries if isinstance(x, dict) and x.get("reason")}
    return reasons, str(body.get("message") or "")


class YouTubeCollector(Collector):
    source = SOURCE
    config_key = "youtube_video_ids"

    def __init__(self, settings: Settings, limit: int | None = None,
                 known_ids: Collection[str] = frozenset()):
        super().__init__(settings, limit, known_ids)
        self.cfg = settings.collectors.youtube
        self.min_interval_s = self.cfg.min_interval_s
        self.api_key = os.environ.get(ENV_KEY, "").strip()
        self._title = ""  # title of the video being collected, added by finish()

    def available(self) -> bool:
        if not self.api_key:
            log.warning("%s: %s is not set in .env, skipping", SOURCE, ENV_KEY)
        return bool(self.api_key)

    def fetch(self, target: str, niche_tag: str) -> Iterator[RawSignal]:
        vid = video_id(target)
        videos = self.call(self._get, "videos", {"part": "snippet", "id": vid})
        if not videos.get("items"):
            raise SkipTarget("video not found (private, deleted or wrong id)")
        snippet = videos["items"][0].get("snippet") or {}
        self._title = scrub(html.unescape(str(snippet.get("title") or ""))).strip()
        owner = snippet.get("channelId")
        log.info("%s: %s = %r", SOURCE, vid, self._title)
        params = {"part": "snippet,replies" if self.cfg.include_replies else "snippet",
                  "videoId": vid, "maxResults": 100, "order": self.cfg.order,
                  "textFormat": "plainText"}
        for _ in range(self.cfg.max_pages_per_video):
            page = self.call(self._get, "commentThreads", params)
            for signal in parse_threads(page, vid, niche_tag, owner, self.cfg.include_replies):
                if signal.external_id not in self.known_ids:
                    yield signal
            if not page.get("nextPageToken"):
                return
            params = {**params, "pageToken": page["nextPageToken"]}

    def finish(self, signal: RawSignal) -> RawSignal:
        if self._title:
            signal.text = f"Video: {self._title}\n\n{signal.text}"
        return signal

    def _get(self, resource: str, params: dict[str, Any]) -> dict[str, Any]:
        """One API call. Key, quota and API-disabled errors stop the source; other 4xx (comments
        disabled, video not found) skip the video; 5xx and network errors are retried."""
        url = f"{self.cfg.api_url.rstrip('/')}/{resource}?{urlencode(params)}"
        request = Request(url, headers={"X-Goog-Api-Key": self.api_key, "User-Agent": USER_AGENT,
                                        "Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.cfg.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            reasons, message = _error(e)
            detail = f"HTTP {e.code} {','.join(sorted(reasons)) or e.reason}: {message}"[:300]
            detail = detail.replace(self.api_key, "[key]")
            if reasons & _STOP_REASONS or e.code == 401 or (e.code == 400 and "API key" in message):
                raise StopSource(detail) from e
            if 400 <= e.code < 500 and e.code != 429:
                raise SkipTarget(detail) from e
            raise
