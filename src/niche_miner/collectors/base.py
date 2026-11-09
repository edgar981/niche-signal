"""Collector ABC: per-target quotas, retry with exponential backoff, per-source rate limit."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, ClassVar, TypeVar

from ..models import RawSignal
from ..settings import Settings
from .filters import TextFilter

log = logging.getLogger(__name__)
T = TypeVar("T")


class SkipTarget(Exception):
    """One target (app id, query, URL) can't be collected — log it and move on, no retry."""


class StopSource(Exception):
    """The site told us to stop (e.g. HTTP 403) — skip the rest of this source's targets."""


def hash_author(source: str, author_id: str) -> str:
    return hashlib.sha256(f"{source}:{author_id}".encode()).hexdigest()


class Collector(ABC):
    source: ClassVar[str]
    config_key: ClassVar[str]  # the only niches.yaml key this collector reads
    min_interval_s: float = 1.0

    def __init__(self, settings: Settings, limit: int | None = None):
        self.settings = settings
        self.limit = limit
        self.filter = TextFilter(settings.filters)
        self._last_request = 0.0

    def available(self) -> bool:
        """Override when the source needs credentials; return False to log and skip."""
        return True

    @abstractmethod
    def fetch(self, target: str, niche_tag: str) -> Iterator[RawSignal]:
        """Yield candidate signals for one target, newest first. Use self.call() for requests."""

    def collect(self, niche_cfg: dict[str, Any]) -> list[RawSignal]:
        """Signals that pass the shared filters. --limit caps the total for this source: each
        target gets an even share of what's left, so small targets don't waste the budget."""
        tag = niche_cfg["tag"]
        targets = [t for t in niche_cfg.get(self.config_key) or [] if t]
        if not targets:
            log.info("%s: no %s configured for %s, skipping", self.source, self.config_key, tag)
            return []
        if not self.available():
            return []
        kept: list[RawSignal] = []
        for i, target in enumerate(targets):
            quota = None
            if self.limit:
                remaining = self.limit - len(kept)
                if remaining <= 0:
                    break
                quota = math.ceil(remaining / (len(targets) - i))
            try:
                kept.extend(self._collect_target(target, tag, quota))
            except SkipTarget as e:
                log.warning("%s: skipping %s: %s", self.source, target, e)
            except StopSource as e:
                log.error("%s: stopping this source: %s", self.source, e)
                break
            except Exception as e:  # retries exhausted; keep going with the next target
                log.error("%s: giving up on %s: %s: %s", self.source, target, type(e).__name__, e)
        if self.filter.dropped:
            dropped = ", ".join(f"{k} {v}" for k, v in self.filter.dropped.most_common())
            log.info("%s: dropped by filters: %s", self.source, dropped)
        return kept[: self.limit] if self.limit else kept

    def _collect_target(self, target: str, tag: str, quota: int | None) -> list[RawSignal]:
        out: list[RawSignal] = []
        for signal in self.fetch(target, tag):
            if self.filter.keep(signal):
                out.append(signal)
                if quota and len(out) >= quota:
                    break
        log.info("%s: %s -> %d signals kept", self.source, target, len(out))
        return out

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Rate-limited request with exponential backoff. SkipTarget/StopSource are not retried."""
        retries = self.settings.collectors.retries
        for attempt in range(retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                return fn(*args, **kwargs)
            except (SkipTarget, StopSource):
                raise
            except Exception as e:
                if attempt == retries:
                    raise
                delay = self.settings.collectors.backoff_s * 2**attempt
                log.warning("%s: %s: %s (attempt %d/%d), retrying in %.0fs", self.source,
                            type(e).__name__, e, attempt + 1, retries + 1, delay)
                time.sleep(delay)
        raise AssertionError("unreachable")
