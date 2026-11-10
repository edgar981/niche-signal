"""Posts (and their top comments) from the public facebook_groups, via an Apify actor
(default in settings.yaml: apify/facebook-groups-scraper). Only public groups can be read.

Authors are only hashed (their profile id); names, profile URLs and pictures are never read.
Posts get the group's name as a first line, and comments an excerpt of the post they answer,
both added after filtering."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..models import RawSignal
from ..settings import ApifyActor
from .apify_common import ApifyCollector, one_line, parse_date, usable_id
from .base import hash_author
from .pii import scrub

SOURCE = "apify_facebook"
POST_EXCERPT_CHARS = 300


def _signal(niche_tag: str, external_id: str, url: Any, author: str | None, text: str,
            posted: Any) -> RawSignal:
    return RawSignal(
        niche_tag=niche_tag,
        source=SOURCE,
        external_id=external_id,
        url=str(url) if url else None,
        author_hash=hash_author(SOURCE, f"user:{author}" if author else external_id),
        author_type="unknown",
        text=scrub(text),
        posted_at=parse_date(posted),
    )


def parse_posts(items: list[dict[str, Any]], niche_tag: str,
                include_comments: bool = True) -> Iterator[tuple[RawSignal, str]]:
    """(signal, context) for each post with text and, optionally, each top comment."""
    for post in items:
        post_id = post.get("legacyId") or post.get("id")
        text = str(post.get("text") or "").strip()
        if not post_id:
            continue
        group = one_line(scrub(str(post.get("groupTitle") or "")), 120)
        if text:
            user = post.get("user") or {}
            author = usable_id(user.get("id") if isinstance(user, dict) else None)
            context = f"Grupo de Facebook: {group}" if usable_id(group) else ""
            yield _signal(niche_tag, f"post:{post_id}", post.get("url"), author, text,
                          post.get("time")), context
        if not include_comments:
            continue
        excerpt = one_line(scrub(text), POST_EXCERPT_CHARS)
        for comment in post.get("topComments") or []:
            comment_id = comment.get("id")
            comment_text = str(comment.get("text") or "").strip()
            if not comment_id or not comment_text:
                continue
            context = f"Comentario a esta publicación de un grupo de Facebook: {excerpt}" \
                if excerpt else ""
            yield _signal(niche_tag, f"comment:{comment_id}", comment.get("commentUrl"),
                          usable_id(comment.get("profileId")), comment_text,
                          comment.get("date")), context


class ApifyFacebookCollector(ApifyCollector):
    source = SOURCE
    config_key = "facebook_groups"

    def actors(self) -> list[ApifyActor]:
        return [self.apify.facebook.run]

    def fetch(self, target: str, niche_tag: str) -> Iterator[RawSignal]:
        items = self.run_actor(self.apify.facebook.run, {"target": target.strip()})
        for signal, context in parse_posts(items, niche_tag, self.apify.facebook.include_comments):
            if signal.external_id in self.known_ids:
                continue
            if context:
                self._context[signal.external_id] = context
            yield signal
