"""LLM pain extraction: unprocessed raw signals → validated ExtractedPain rows, 20 per request.

Each batch is sent once; items that come back missing or invalid are retried once together,
and anything still failing is stored as a failed row (rerun with `extract --retry-failed`)."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from . import db
from .llm import LLMError
from .models import ExtractedPain
from .settings import LLMSettings
from .source_rules import country_rule, source_hint

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You analyze public texts written in Spanish, mostly by people in Colombia (app-store reviews, \
marketplace questions, video comments, job posts, social-media posts), to find unmet needs \
that a small software or service business could solve. The texts are often informal, with \
typos and slang.

You receive a JSON array of items, each with an "id", a "fuente" (where it was posted) and a \
"texto". Return exactly one result per item, with the same id. Fields:

- is_complaint: true if the text expresses a problem, a frustration, an unmet need, or a \
workaround the person is putting up with. false for praise, thanks, ads, spam, jokes, or \
neutral questions with no problem behind them.
- pain: one sentence in Spanish naming the underlying unmet need in general terms (what the \
person needs and cannot get) instead of retelling the anecdote. Example: "Necesita emitir \
facturas electrónicas sin que la DIAN las rechace por errores de validación." null when \
is_complaint is false.
- pain_type: null when is_complaint is false; otherwise one of
  "missing_capability": no tool they have can do the job (e.g. no way to invoice from the \
phone, no bank reconciliation);
  "incumbent_quality": the tool or provider they already use does it badly: bugs, crashes, \
slowness, broken updates, lost data, poor support;
  "pricing": cost, plan limits, billing errors, double charges, refunds, unclear subscriptions;
  "onboarding": signing up, setting it up, learning to use it, or migrating data into it.
- who: the writer's concrete role or business, in Spanish, e.g. "contador independiente", \
"dueña de ferretería", "auxiliar contable", "comerciante", "administrador de restaurante". \
Infer it from the texto and the fuente when the clues support it; what they do is a clue \
(someone who issues electronic invoices or pays an accountant is at least a "dueño de \
negocio"). Never a generic label such as "usuario de <app>" or just "cliente". null if there \
is no clue.
- who_type: "business" if they write as a business, a professional, or a worker doing their \
job (issuing invoices, paying an accountant, payroll, inventory or running a store all mean \
business); "consumer" if they write as a private person or end customer; "unknown" if you \
cannot tell.
- country: where the writer is: "co", "mx", "pe", "ar", "es", "other" or "unknown". Use \
explicit mentions and strong cues: tax authority (DIAN = co, SAT or CFDI = mx, SUNAT = pe, \
AFIP or ARCA = ar, Hacienda or AEAT = es), currency (euros = es), cities, regional slang. A \
store's country does not prove the writer's country. Google Maps and Google Play listings mix \
countries: for those, "co" needs an explicit Colombian cue in the texto (a Colombian city, \
DIAN, COP, pesos, NIT, RUT, Nequi, PSE); never infer it from the store, the app or the locale. \
"unknown" if there is no cue.
- current_tool: the product they use today for this job, by product name only, e.g. "Siigo \
Nube", "Alegra", "Loggro" (for an app review, usually the reviewed app). Never a package id \
such as "com.example.app", and no extra words like "app" or "móvil". "manual/Excel/WhatsApp" \
when they do the job by hand, in spreadsheets or in chats. null if nothing is stated or implied.
- spend_signal: a quote copied exactly from the texto, character for character, only if it \
shows money paid to a provider, a person or a tool ("le pago 300 mil al mes a un contador", \
"me cobraron 8 millones por la renovación"), recurring hours lost ("pierdo dos días cada \
cierre"), or someone hired or paid to deal with it. Tax owed, fines or penalties and lost \
sales are NOT spend (they go in cost_of_problem). Workarounds (reinstalling, logging in \
again), one-off waits ("llevo un mes esperando") and complaints that something is expensive \
are not spend signals either. Do not translate, shorten or fix typos. null if there is none.
- cost_of_problem: a quote copied exactly from the texto, with the same copying rules, only if \
it shows what the problem costs them that is not spend: tax owed, fines or penalties, \
interest, lost sales or customers ("me tocó pagar una sanción de 2 millones", "perdí ventas \
por no poder facturar"). null if there is none.
- severity: 1 to 5. 1 = minor annoyance; 3 = recurring problem that costs time or money; \
5 = it blocks their work or business, or makes them lose money or customers.

Job posts (fuente "oferta de empleo"): the hiring business is the writer, and paying a person \
to do the work is its workaround. is_complaint is true when the post describes recurring \
operational work that software could take over or speed up (bookkeeping, bank \
reconciliation, invoicing, data entry, inventory, collections, payroll, Excel reports); pain \
names that work as the business's need, e.g. "Necesita conciliar los extractos bancarios y \
registrar las facturas de proveedores cada mes."; pain_type is usually "missing_capability"; \
who is the kind of business hiring, from its sector or the text (e.g. "empresa \
distribuidora", "constructora"), never the job title; who_type is "business"; current_tool is \
the software the post asks for, by product name ("Siigo", "World Office"), or \
"manual/Excel/WhatsApp" if it only asks for Excel; spend_signal is the "Salario:" line or the \
sentence that shows the hire, copied exactly; cost_of_problem is null; severity is usually 3. is_complaint is false \
for posts without such work (sales, driving, cleaning, customer service) and for courses or \
ads posing as jobs.

Do not invent facts the text does not support."""

_NULLABLE_STRING = {"type": ["string", "null"]}
PAIN_TYPES = ["missing_capability", "incumbent_quality", "pricing", "onboarding"]
COUNTRIES = ["co", "mx", "pe", "ar", "es", "other", "unknown"]
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "is_complaint": {"type": "boolean"},
                    "pain": _NULLABLE_STRING,
                    "pain_type": {"anyOf": [{"type": "string", "enum": PAIN_TYPES},
                                            {"type": "null"}]},
                    "who": _NULLABLE_STRING,
                    "who_type": {"type": "string", "enum": ["business", "consumer", "unknown"]},
                    "country": {"type": "string", "enum": COUNTRIES},
                    "current_tool": _NULLABLE_STRING,
                    "spend_signal": _NULLABLE_STRING,
                    "cost_of_problem": _NULLABLE_STRING,
                    "severity": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                },
                "required": ["id", "is_complaint", "pain", "pain_type", "who", "who_type",
                             "country", "current_tool", "spend_signal", "cost_of_problem",
                             "severity"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


class Completer(Protocol):
    def complete(self, *, purpose: str, model: str, system: str, user: str,
                 schema: dict[str, Any] | None = None, n_items: int = 0,
                 max_tokens: int | None = None, effort: str | None = None) -> str: ...


@dataclass
class ExtractStats:
    signals: int = 0
    complaints: int = 0
    failed: int = 0
    retried: int = 0
    spend_dropped: int = 0  # spend / cost quotes discarded because they weren't verbatim
    country_reset: int = 0  # Maps/Play "co" without an explicit Colombian cue -> "unknown"


def build_prompt(rows: list[sqlite3.Row], max_chars: int) -> str:
    """Batch-local ids (0..n-1) keep the response short; they're mapped back to DB ids."""
    items = [{"id": i, "fuente": source_hint(r["source"], r["url"]), "texto": r["text"][:max_chars]}
             for i, r in enumerate(rows)]
    return "Analiza cada texto:\n" + json.dumps(items, ensure_ascii=False)


def parse_results(text: str, n: int) -> tuple[dict[int, ExtractedPain], dict[int, str]]:
    """Validate each result on its own so one bad item can't sink the batch.
    Returns (valid results, errors) keyed by batch index; raises ValueError on non-JSON."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"response is not valid JSON ({e})") from e
    results = data.get("results") if isinstance(data, dict) else data
    if not isinstance(results, list):
        raise ValueError("response JSON has no 'results' list")
    ok: dict[int, ExtractedPain] = {}
    errors: dict[int, str] = {}
    for obj in results:
        idx = obj.get("id") if isinstance(obj, dict) else None
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < n or idx in ok:
            continue
        try:
            ok[idx] = ExtractedPain.model_validate(obj)
            errors.pop(idx, None)
        except ValidationError as e:
            errors[idx] = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'item'}: {err['msg']}" for err in e.errors())
    for i in range(n):
        if i not in ok:
            errors.setdefault(i, "missing from response")
    return ok, errors


_QUOTE_CHARS = "\"'“”‘’«» .,;:"


def _normalize(s: str) -> str:
    return " ".join(s.casefold().split())


def is_verbatim(quote: str, text: str) -> bool:
    """Every fragment of the quote (split on ellipses) occurs in the text, ignoring case,
    whitespace and surrounding quotes/punctuation."""
    haystack = _normalize(text)
    fragments = [_normalize(f).strip(_QUOTE_CHARS) for f in re.split(r"\.\.\.|…", quote)]
    fragments = [f for f in fragments if f]
    return bool(fragments) and all(f in haystack for f in fragments)


class Extractor:
    def __init__(self, conn: sqlite3.Connection, llm: Completer, settings: LLMSettings,
                 model: str | None = None):
        self.conn = conn
        self.llm = llm
        self.settings = settings
        self.model = model or settings.extract_model

    def run(self, niche: str | None = None, limit: int | None = None,
            retry_failed: bool = False, redo: bool = False) -> ExtractStats:
        """Extract pending signals. retry_failed re-runs failures; redo re-runs everything
        (e.g. after a prompt change) and drops the cluster memberships of the old results."""
        if redo or retry_failed:
            cleared = db.delete_extractions(self.conn, niche, failed_only=not redo)
            log.info("cleared %d %sextractions to re-run", cleared, "" if redo else "failed ")
        pending = db.pending_signals(self.conn, niche, limit)
        stats = ExtractStats()
        size = self.settings.batch_size
        for start in range(0, len(pending), size):
            self._batch(pending[start:start + size], stats)
            log.info("extract: %d/%d signals done", min(start + size, len(pending)), len(pending))
        return stats

    def _batch(self, rows: list[sqlite3.Row], stats: ExtractStats) -> None:
        ok, errors = self._attempt(rows)
        if errors:  # one retry, only for the items that failed
            retry = sorted(errors)
            stats.retried += len(retry)
            ok2, errors2 = self._attempt([rows[i] for i in retry])
            for j, i in enumerate(retry):
                if j in ok2:
                    ok[i] = ok2[j]
                    del errors[i]
                else:
                    errors[i] = errors2.get(j, errors[i])
        results: list[tuple[int, ExtractedPain | None, str | None]] = []
        for i, row in enumerate(rows):
            pain = ok.get(i)
            for field in ("spend_signal", "cost_of_problem"):
                quote = getattr(pain, field) if pain else None
                if pain and quote and not is_verbatim(quote, row["text"]):
                    log.debug("dropping non-verbatim %s for signal %s: %r", field, row["id"], quote)
                    pain = pain.model_copy(update={field: None})
                    stats.spend_dropped += 1
            country = country_rule(row["source"], row["text"], pain.country) if pain else None
            if pain and country != pain.country:
                pain = pain.model_copy(update={"country": country})
                stats.country_reset += 1
            results.append((row["id"], pain, None if pain else errors.get(i)))
            stats.signals += 1
            stats.complaints += bool(pain and pain.is_complaint)
            stats.failed += pain is None
        db.save_pains(self.conn, results, self.model)

    def _attempt(self, rows: list[sqlite3.Row]) -> tuple[dict[int, ExtractedPain], dict[int, str]]:
        try:
            text = self.llm.complete(
                purpose="extract", model=self.model, system=SYSTEM_PROMPT,
                user=build_prompt(rows, self.settings.max_text_chars),
                schema=RESPONSE_SCHEMA, n_items=len(rows))
            return parse_results(text, len(rows))
        except (LLMError, ValueError) as e:
            log.warning("extract: batch of %d failed: %s", len(rows), e)
            return {}, {i: str(e) for i in range(len(rows))}
