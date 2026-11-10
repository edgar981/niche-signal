"""Shared plumbing for the Apify collectors: token check, input placeholders, one capped run.

Actor ids and inputs come from config (collectors.apify in settings.yaml), so an actor can be
swapped without code changes as long as its output has the same shape. Without APIFY_TOKEN in
.env the source is logged and skipped.

Every run is started with a cost cap (max_total_charge_usd) and, for pay-per-item actors, an
item cap (max_items, also lowered by --limit). Apify charges for every item an actor returns,
including ones already in the DB, so keep the caps small. A run is never restarted on error:
apify-client already retries the HTTP calls, and a second run would be charged again."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from apify_client import ApifyClient
from apify_client.errors import ApifyApiError

from ..models import RawSignal
from ..settings import ApifyActor, Settings
from .base import Collector, SkipTarget, StopSource

log = logging.getLogger(__name__)

ENV_KEY = "APIFY_TOKEN"
_LINK = re.compile(r"https?://\S+")


def render(value: Any, variables: dict[str, Any]) -> Any:
    """Fill placeholders in a config input. A string that is exactly "{name}" becomes the
    variable itself (so "{video_urls}" can become a list); "{name}" inside a longer string is
    replaced by the variable's text."""
    if isinstance(value, dict):
        return {k: render(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, variables) for v in value]
    if isinstance(value, str):
        if value.startswith("{") and value.endswith("}") and value[1:-1] in variables:
            return variables[value[1:-1]]
        for name, var in variables.items():
            if isinstance(var, str):
                value = value.replace("{" + name + "}", var)
    return value


def parse_date(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value)).astimezone(UTC)
    except ValueError:
        return None


def one_line(text: str, max_chars: int = 200) -> str:
    """Context for finish(): a single line, so the model can tell it apart from the text."""
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def usable_id(value: Any) -> str | None:
    """An author id worth hashing: not empty and not a masked placeholder like '******'."""
    text = str(value or "").strip()
    return text if text and set(text) != {"*"} else None


class ApifyCollector(Collector):
    min_interval_s = 0.0  # Apify runs the actor; there is no site of ours to rate-limit

    def __init__(self, settings: Settings, limit: int | None = None,
                 known_ids: Collection[str] = frozenset()):
        super().__init__(settings, limit, known_ids)
        self.apify = settings.collectors.apify
        self.token = os.environ.get(ENV_KEY, "").strip()
        self.cost_usd = 0.0  # summed over this collector's runs, for the log
        self._context: dict[str, str] = {}  # external id -> first line added by finish()

    def actors(self) -> list[ApifyActor]:
        """The actor configs this source needs; all must name an actor."""
        raise NotImplementedError

    def available(self) -> bool:
        if not self.token:
            log.warning("%s: %s is not set in .env, skipping", self.source, ENV_KEY)
            return False
        if not all(spec.actor for spec in self.actors()):
            log.warning("%s: no actor configured under collectors.apify in settings.yaml, "
                        "skipping", self.source)
            return False
        return True

    def collect(self, niche_cfg: dict[str, Any]) -> list[RawSignal]:
        kept = super().collect(niche_cfg)
        if self.cost_usd:  # Apify settles usage after a run ends; the console has the final figure
            log.info("%s: Apify cost this run about $%.4f (final figure in the Apify console)",
                     self.source, self.cost_usd)
        return kept

    def finish(self, signal: RawSignal) -> RawSignal:
        if context := self._context.get(signal.external_id):
            signal.text = f"{context}\n\n{signal.text}"
        return signal

    def run_actor(self, spec: ApifyActor, variables: dict[str, Any]) -> list[dict[str, Any]]:
        """Run one actor to completion and return its dataset items (also after a failed or
        capped run: whatever it produced was paid for)."""
        caps = [x for x in (spec.max_items, self.limit) if x]
        max_items = min(caps) if caps else None
        cap = spec.max_charge_usd or self.apify.max_charge_usd
        client = ApifyClient(self.token)
        try:
            run = client.actor(spec.actor).start(
                run_input=render(spec.input, variables), max_items=max_items,
                max_total_charge_usd=Decimal(str(cap)),
                run_timeout=timedelta(seconds=self.apify.run_timeout_s))
            log.info("%s: started %s (run %s, cap $%.2f, max %s items)", self.source,
                     spec.actor, run.id, cap, max_items or "no")
            done = client.run(run.id).wait_for_finish(
                wait_duration=timedelta(seconds=self.apify.run_timeout_s + 120)) or run
            items = list(client.dataset(done.default_dataset_id).iterate_items(clean=True))
            # Usage is settled a little after the run ends; read it again once items are in.
            done = client.run(done.id).get() or done
        except ApifyApiError as e:
            # Same failure for every target (bad token, no credit, wrong actor id): stop the
            # source. Other 4xx (e.g. an input the actor rejects for this target): skip it.
            code = getattr(e, "status_code", 0) or 0
            kind = str(getattr(e, "type", "") or "")
            if code in (401, 402, 403, 404) or "limit" in kind or "credit" in kind:
                raise StopSource(self._detail(e, f"{spec.actor}: HTTP {code} {kind}")) from e
            if 400 <= code < 500:
                raise SkipTarget(self._detail(e, f"HTTP {code} {kind}")) from e
            raise
        self.cost_usd += float(done.usage_total_usd or 0)
        status = _LINK.sub("[link]", done.status_message or "")  # may hold signed URLs
        level = logging.INFO if done.status == "SUCCEEDED" else logging.WARNING
        log.log(level, "%s: %s run %s %s%s: %d items, about $%.4f so far", self.source,
                spec.actor, done.id, done.status, f" ({status})" if status else "", len(items),
                float(done.usage_total_usd or 0))
        return items

    def _detail(self, e: Exception, prefix: str) -> str:
        text = f"{prefix}: {getattr(e, 'message', None) or e}"[:300]
        return text.replace(self.token, "[token]") if self.token else text
