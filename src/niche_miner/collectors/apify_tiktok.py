"""TikTok comments for the tiktok_hashtags, via two Apify actors (defaults in settings.yaml):
one finds the hashtag's videos (clockworks/free-tiktok-scraper), the other reads their comments
(clockworks/tiktok-comments-scraper, given "{video_urls}").

Commenters are only hashed (their numeric user id); usernames and avatars are never read.
Comments by the video's own author are left out. The video's caption is added as a first line
after filtering, so the model knows what a comment answers."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, NamedTuple

from ..models import RawSignal
from ..settings import ApifyActor
from .apify_common import ApifyCollector, one_line, parse_date, usable_id
from .base import SkipTarget, hash_author
from .pii import scrub

SOURCE = "apify_tiktok"
_VIDEO_ID = re.compile(r"/video/(\d+)")


class Video(NamedTuple):
    url: str
    caption: str
    author_id: str | None


def video_key(url: Any) -> str | None:
    """The numeric video id inside a TikTok URL, so URL variants still match."""
    match = _VIDEO_ID.search(str(url or ""))
    return match.group(1) if match else None


def parse_videos(items: list[dict[str, Any]]) -> dict[str, Video]:
    """Video id -> Video, skipping error items ("Post not found or private")."""
    videos: dict[str, Video] = {}
    for item in items:
        url = item.get("webVideoUrl")
        key = video_key(url)
        if not key or item.get("error"):
            continue
        author = item.get("authorMeta") or {}
        videos[key] = Video(str(url), one_line(scrub(str(item.get("text") or ""))),
                            usable_id(author.get("id") if isinstance(author, dict) else None))
    return videos


def parse_comments(items: list[dict[str, Any]], niche_tag: str,
                   videos: dict[str, Video]) -> Iterator[RawSignal]:
    for item in items:
        comment_id, text = item.get("cid"), str(item.get("text") or "").strip()
        video = videos.get(video_key(item.get("videoWebUrl")) or "")
        user = usable_id(item.get("uid"))
        if not comment_id or not text or (video and user and user == video.author_id):
            continue
        yield RawSignal(
            niche_tag=niche_tag,
            source=SOURCE,
            external_id=str(comment_id),
            url=item.get("videoWebUrl") or (video.url if video else None),
            author_hash=hash_author(SOURCE, f"user:{user}" if user else f"comment:{comment_id}"),
            author_type="unknown",
            text=scrub(text),
            posted_at=parse_date(item.get("createTimeISO")),
        )


class ApifyTikTokCollector(ApifyCollector):
    source = SOURCE
    config_key = "tiktok_hashtags"

    def actors(self) -> list[ApifyActor]:
        return [self.apify.tiktok.videos, self.apify.tiktok.comments]

    def fetch(self, target: str, niche_tag: str) -> Iterator[RawSignal]:
        hashtag = target.strip().lstrip("#")
        videos = parse_videos(self.run_actor(self.apify.tiktok.videos, {"target": hashtag}))
        if not videos:
            raise SkipTarget(f"no videos found for #{hashtag}")
        items = self.run_actor(self.apify.tiktok.comments,
                               {"target": hashtag, "video_urls": [v.url for v in videos.values()]})
        for signal in parse_comments(items, niche_tag, videos):
            if signal.external_id in self.known_ids:
                continue
            video = videos.get(video_key(signal.url) or "")
            if video and video.caption:
                self._context[signal.external_id] = f"Video de TikTok: {video.caption}"
            yield signal
