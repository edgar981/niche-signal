"""Google Maps reviews for the gmaps_queries, via an Apify actor (default in settings.yaml:
compass/crawler-google-places, which searches places and returns each with its reviews).

The actor is asked not to fetch reviewers' personal data at all (scrapeReviewsPersonalData:
false in config); if a reviewer id still comes back it is only hashed, and names are never
read. Only reviews with at most collectors.apify.gmaps.max_stars stars are kept; owner replies
are left out. The place's name and category are added as a first line after filtering."""

from __future__ import annotations

import html
import re
from collections.abc import Iterator
from typing import Any

from ..models import RawSignal
from ..settings import ApifyActor
from .apify_common import ApifyCollector, one_line, parse_date, usable_id
from .base import hash_author
from .pii import scrub

SOURCE = "apify_gmaps"
_BR = re.compile(r"<br\s*/?>", re.I)


def review_text(value: Any) -> str:
    """Review text arrives with HTML line breaks and entities ("<br><br>", "&amp;")."""
    return html.unescape(_BR.sub("\n", str(value or ""))).strip()


def place_context(place: dict[str, Any]) -> str:
    title = one_line(scrub(str(place.get("title") or "")), 120)
    category = one_line(str(place.get("categoryName") or ""), 60)
    if not title:
        return ""
    return f"Lugar en Google Maps: {title}" + (f" ({category})" if category else "")


def parse_places(items: list[dict[str, Any]], niche_tag: str,
                 max_stars: int) -> Iterator[tuple[RawSignal, str]]:
    """(signal, place context) for each low-star review with text, across all places."""
    for place in items:
        context = place_context(place)
        for review in place.get("reviews") or []:
            stars, review_id = review.get("stars"), review.get("reviewId")
            text = review_text(review.get("text"))
            if not review_id or not text or not isinstance(stars, int | float) or stars > max_stars:
                continue
            reviewer = usable_id(review.get("reviewerId"))
            yield RawSignal(
                niche_tag=niche_tag,
                source=SOURCE,
                external_id=str(review_id),
                url=review.get("reviewUrl") or place.get("url"),
                author_hash=hash_author(SOURCE, f"reviewer:{reviewer}" if reviewer
                                        else f"review:{review_id}"),
                author_type="unknown",
                text=scrub(text),
                posted_at=parse_date(review.get("publishedAtDate")),
            ), context


class ApifyGmapsCollector(ApifyCollector):
    source = SOURCE
    config_key = "gmaps_queries"

    def actors(self) -> list[ApifyActor]:
        return [self.apify.gmaps.run]

    def fetch(self, target: str, niche_tag: str) -> Iterator[RawSignal]:
        items = self.run_actor(self.apify.gmaps.run, {"target": target.strip()})
        for signal, context in parse_places(items, niche_tag, self.apify.gmaps.max_stars):
            if signal.external_id in self.known_ids:
                continue
            if context:
                self._context[signal.external_id] = context
            yield signal
