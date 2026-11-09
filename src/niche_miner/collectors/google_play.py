"""Google Play reviews via google-play-scraper: Colombian store, Spanish, 1–2 stars by default."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from google_play_scraper import Sort, app, reviews
from google_play_scraper.exceptions import NotFoundError

from ..models import RawSignal
from ..settings import GooglePlaySettings, Settings
from .base import Collector, SkipTarget, hash_author

log = logging.getLogger(__name__)

SOURCE = "google_play"
# Display names Google shows for reviewers who hide their name — not a stable author id.
ANONYMOUS_NAMES = {"a google user", "un usuario de google", "usuario de google"}


def review_url(app_id: str, review_id: str, cfg: GooglePlaySettings) -> str:
    return (f"https://play.google.com/store/apps/details?id={app_id}"
            f"&hl={cfg.lang}&gl={cfg.country.upper()}&reviewId={review_id}")


def _to_utc(value: Any) -> datetime | None:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC)  # the library returns naive local time (fromtimestamp)
    return None


def parse_review(raw: dict[str, Any], app_id: str, niche_tag: str,
                 cfg: GooglePlaySettings) -> RawSignal | None:
    """Map one google-play-scraper review dict to a RawSignal, or None if it doesn't qualify.
    The reviewer's name is only hashed; replyContent (which repeats the name) is ignored."""
    review_id = raw.get("reviewId")
    text = (raw.get("content") or "").strip()
    score = raw.get("score")
    if not review_id or not text or not isinstance(score, int) or score > cfg.max_score:
        return None
    name = (raw.get("userName") or "").strip()
    author_key = review_id if not name or name.casefold() in ANONYMOUS_NAMES else name
    return RawSignal(
        niche_tag=niche_tag,
        source=SOURCE,
        external_id=review_id,
        url=review_url(app_id, review_id, cfg),
        author_hash=hash_author(SOURCE, author_key),
        author_type="unknown",
        text=text,
        posted_at=_to_utc(raw.get("at")),
    )


class GooglePlayCollector(Collector):
    source = SOURCE
    config_key = "google_play_apps"

    def __init__(self, settings: Settings, limit: int | None = None):
        super().__init__(settings, limit)
        self.cfg = settings.collectors.google_play
        self.min_interval_s = self.cfg.min_interval_s

    def fetch(self, app_id: str, niche_tag: str) -> Iterator[RawSignal]:
        self.call(self._check_app, app_id)
        per_score = max(1, self.cfg.max_reviews_per_app // self.cfg.max_score)
        for score in range(1, self.cfg.max_score + 1):
            token, fetched = None, 0
            while fetched < per_score:
                batch, token = self.call(self._page, app_id, score, token)
                for raw in batch[: per_score - fetched]:
                    if signal := parse_review(raw, app_id, niche_tag, self.cfg):
                        yield signal
                fetched += len(batch)
                if len(batch) < self.cfg.page_size:
                    break  # no more reviews with this score

    def _check_app(self, app_id: str) -> None:
        try:
            info = app(app_id, lang=self.cfg.lang, country=self.cfg.country)
        except NotFoundError as e:
            raise SkipTarget(f"app id not found in the {self.cfg.country.upper()} store") from e
        log.info("%s: %s = %r (%s ratings)", SOURCE, app_id, info.get("title"), info.get("ratings"))

    def _page(self, app_id: str, score: int, token: Any) -> tuple[list[dict[str, Any]], Any]:
        return reviews(app_id, lang=self.cfg.lang, country=self.cfg.country, sort=Sort.NEWEST,
                       count=self.cfg.page_size, filter_score_with=score, continuation_token=token)
