"""Shared test helpers (pytest puts tests/ on sys.path, so test modules import this directly)."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from niche_miner import db
from niche_miner.models import Cluster, ExtractedPain, RawSignal


def make_signal(i: int, *, niche: str = "test_niche", source: str = "google_play",
                author: str = "author", text: str | None = None,
                author_type: str = "unknown") -> RawSignal:
    return RawSignal(niche_tag=niche, source=source, external_id=f"ext-{i}",
                     url=f"https://example.com/review/{i}", author_hash=f"hash-{author}",
                     author_type=author_type,
                     text=text or f"Texto de prueba número {i}: una queja suficientemente larga.")


def good_result(i: int, **overrides: Any) -> dict[str, Any]:
    result = {"id": i, "is_complaint": True, "pain": f"Necesita resolver el problema {i}.",
              "pain_type": "missing_capability", "who": "contador independiente",
              "who_type": "business", "country": "co", "current_tool": "Siigo",
              "spend_signal": None, "severity": 3}
    result.update(overrides)
    return result


def response(*results: dict[str, Any]) -> str:
    return json.dumps({"results": list(results)}, ensure_ascii=False)


def prompt_items(call: dict[str, Any]) -> list[dict[str, Any]]:
    """The batch items one FakeLLM call was asked to extract."""
    return json.loads(call["user"].split("\n", 1)[1])


def echo_valid(call: dict[str, Any]) -> str:
    return response(*(good_result(item["id"]) for item in prompt_items(call)))


class FakeLLM:
    """Stands in for llm.LLM. Each queued reply is a string, an Exception to raise, or a
    callable taking the call kwargs. Unexpected extra calls fail the test."""

    def __init__(self, *replies: Any):
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        assert self.replies, f"unexpected LLM call: {kwargs.get('purpose')}"
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(kwargs)
        if isinstance(reply, Exception):
            raise reply
        return reply


def seed_pain(conn: sqlite3.Connection, i: int, *, author: str, niche: str = "test_niche",
              who: str | None = "contador", who_type: str = "unknown", tool: str | None = "Siigo",
              spend: str | None = None, text: str | None = None,
              pain_type: str = "missing_capability", country: str = "co") -> int:
    """Insert a raw signal plus a complaint extraction; returns the extracted_pains id."""
    db.insert_signals(conn, [make_signal(i, niche=niche, author=author, text=text)])
    signal_id = conn.execute("SELECT id FROM raw_signals WHERE external_id = ?",
                             (f"ext-{i}",)).fetchone()[0]
    pain = ExtractedPain(is_complaint=True, pain=f"Dolor número {i}", pain_type=pain_type,
                         who=who, who_type=who_type, country=country, current_tool=tool,
                         spend_signal=spend, severity=3)
    db.save_pains(conn, [(signal_id, pain, None)], "fake-model")
    return conn.execute("SELECT id FROM extracted_pains WHERE raw_signal_id = ?",
                        (signal_id,)).fetchone()[0]


def seed_cluster(conn: sqlite3.Connection, pain_ids: list[int], *, label: str = "Etiqueta",
                 niche: str = "test_niche", run_id: str = "run-1", noise: bool = False) -> int:
    members = [(pain_id, 0.1 * k) for k, pain_id in enumerate(pain_ids)]
    cluster = Cluster(label=label, niche_tag=niche, run_id=run_id, is_noise=noise,
                      size=len(pain_ids), representative_ids=pain_ids[:3])
    return db.insert_cluster(conn, cluster, members)
