"""Thin wrapper over the Anthropic SDK: request shape, refusal/truncation checks, cost logging.
Every call is written to the llm_calls table with tokens and USD cost."""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import anthropic

from . import db
from .settings import LLMSettings

log = logging.getLogger(__name__)


def has_credentials() -> bool:
    """An API key/token in the env (.env), or an `ant auth login` profile the SDK reads itself.
    The SDK only fails at request time otherwise, so callers check this up front."""
    if any(os.environ.get(v) for v in
           ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")):
        return True
    return (Path.home() / ".config" / "anthropic").is_dir()


class LLMError(RuntimeError):
    """A response we can't use (refusal, truncation). Callers treat it like a parse failure."""


class LLM:
    def __init__(self, conn: sqlite3.Connection, settings: LLMSettings, run_id: str,
                 client: Any | None = None):
        self.conn = conn
        self.settings = settings
        self.run_id = run_id
        self._client = client
        self._unpriced: set[str] = set()

    @property
    def client(self) -> Any:
        if self._client is None:
            # The SDK retries 408/409/429/5xx and connection errors with backoff.
            self._client = anthropic.Anthropic(max_retries=4)
        return self._client

    def complete(self, *, purpose: str, model: str, system: str, user: str,
                 schema: dict[str, Any] | None = None, n_items: int = 0,
                 max_tokens: int | None = None, effort: str | None = None) -> str:
        """One Messages API call; returns the response text (JSON when a schema is given).
        `effort` overrides llm.effort for this call."""
        output_config: dict[str, Any] = {}
        if effort := effort or self.settings.effort:
            output_config["effort"] = effort
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        extra = {"output_config": output_config} if output_config else {}
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens or self.settings.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **extra,
        )
        usage = response.usage
        db.log_llm_call(self.conn, self.run_id, purpose, model, usage.input_tokens,
                        usage.output_tokens, self.cost(model, usage), n_items)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMError(f"refused ({getattr(details, 'category', None) or 'no category'})")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")
        return "".join(block.text for block in response.content if block.type == "text")

    def cost(self, model: str, usage: Any) -> float | None:
        """USD for one call from config pricing; None (with a warning) for unpriced models."""
        price = self.settings.pricing.get(model)
        if price is None:
            if model not in self._unpriced:
                log.warning("no llm.pricing entry for %s in config/settings.yaml; "
                            "its cost will not be tracked", model)
                self._unpriced.add(model)
            return None
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        input_equiv = usage.input_tokens + 1.25 * cache_write + 0.1 * cache_read
        return (input_equiv * price["input"] + usage.output_tokens * price["output"]) / 1e6
