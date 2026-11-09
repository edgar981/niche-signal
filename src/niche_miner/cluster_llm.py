"""`cluster --method llm`: one call proposes ~15 Spanish themes from all pains, then batched
calls assign each pain to one theme (0 = none fits). Unassigned pains become noise."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from .extract import Completer
from .llm import LLMError
from .settings import ClusterSettings

log = logging.getLogger(__name__)

THEMES_SYSTEM = (
    "Recibes una lista de dolores (necesidades no resueltas) extraídos de quejas escritas por "
    "personas en Colombia. Propón alrededor de {n} temas que agrupen esos dolores por la "
    "necesidad de fondo, no por la app que mencionan. Cada tema tiene un \"label\" en español de "
    "máximo 6 palabras que nombra el problema (no la solución) y una \"description\" de una frase "
    "que dice qué dolores entran. Los temas no se solapan y entre todos cubren la mayoría de los "
    "dolores. No incluyas un tema genérico de \"otros\"."
)
THEMES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"themes": {"type": "array", "items": {
        "type": "object",
        "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
        "required": ["label", "description"], "additionalProperties": False}}},
    "required": ["themes"], "additionalProperties": False,
}
ASSIGN_SYSTEM = (
    "Asigna cada dolor al tema numerado que mejor lo describe. Usa 0 si ningún tema encaja bien. "
    "Devuelve exactamente una asignación por id."
)
_LABEL_STRIP = "\"'“”«»*#.- "


def clean_label(text: str, max_words: int = 10) -> str:
    line = next((ln for ln in text.strip().splitlines() if ln.strip()), "").strip(_LABEL_STRIP)
    line = re.sub(r"^(etiqueta|label)\s*:\s*", "", line, flags=re.I).strip(_LABEL_STRIP)
    return " ".join(line.split()[:max_words]) or "Sin etiqueta"


def _assign_schema(n_themes: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"assignments": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"},
                           "theme": {"type": "integer", "enum": list(range(n_themes + 1))}},
            "required": ["id", "theme"], "additionalProperties": False}}},
        "required": ["assignments"], "additionalProperties": False,
    }


def propose_themes(llm: Completer, model: str, pains: Sequence[str],
                   cfg: ClusterSettings) -> list[tuple[str, str]]:
    """(label, description) pairs, labels deduplicated. Raises LLMError if none come back."""
    text = llm.complete(purpose="themes", model=model, system=THEMES_SYSTEM.format(n=cfg.llm_themes),
                        user="Dolores:\n" + "\n".join(f"- {p}" for p in pains),
                        schema=THEMES_SCHEMA, n_items=len(pains), effort=cfg.llm_theme_effort)
    try:
        raw = json.loads(text).get("themes") or []
    except (ValueError, AttributeError) as e:
        raise LLMError(f"theme proposal is not valid JSON ({e})") from e
    themes: list[tuple[str, str]] = []
    seen: set[str] = set()
    for t in raw:
        label = clean_label(str(t.get("label") or ""))
        if label != "Sin etiqueta" and label.casefold() not in seen:
            seen.add(label.casefold())
            themes.append((label, str(t.get("description") or "").strip()))
    if not themes:
        raise LLMError("theme proposal returned no themes")
    return themes


def _assign_batch(llm: Completer, model: str, theme_block: str, n_themes: int,
                  pains: Sequence[str]) -> dict[int, int]:
    """Batch index → theme number (0 = none). Bad or missing items are simply absent."""
    user = (f"Temas:\n{theme_block}\n\nDolores:\n"
            + json.dumps([{"id": j, "dolor": p} for j, p in enumerate(pains)], ensure_ascii=False))
    try:
        text = llm.complete(purpose="assign", model=model, system=ASSIGN_SYSTEM, user=user,
                            schema=_assign_schema(n_themes), n_items=len(pains))
        items = json.loads(text).get("assignments") or []
    except (LLMError, ValueError, AttributeError) as e:
        log.warning("cluster: assignment batch of %d failed: %s", len(pains), e)
        return {}
    out: dict[int, int] = {}
    for a in items:
        j, k = a.get("id"), a.get("theme")
        valid = all(isinstance(v, int) and not isinstance(v, bool) for v in (j, k))
        if valid and 0 <= j < len(pains) and 0 <= k <= n_themes and j not in out:
            out[j] = k
    return out


def llm_assign(llm: Completer, model: str, pains: Sequence[str],
               cfg: ClusterSettings) -> tuple[np.ndarray, dict[int, str]]:
    """(label per pain: theme index or -1, {theme index: label}). Missing assignments are
    retried once per batch; anything still unassigned is noise."""
    themes = propose_themes(llm, model, pains, cfg)
    block = "\n".join(f"{i}. {label}: {desc}" for i, (label, desc) in enumerate(themes, 1))
    labels = np.full(len(pains), -1)
    for start in range(0, len(pains), cfg.llm_batch_size):
        batch = list(pains[start:start + cfg.llm_batch_size])
        got = _assign_batch(llm, model, block, len(themes), batch)
        missing = [j for j in range(len(batch)) if j not in got]
        if missing:
            retry = _assign_batch(llm, model, block, len(themes), [batch[j] for j in missing])
            got.update({missing[j]: k for j, k in retry.items()})
        for j, k in got.items():
            labels[start + j] = k - 1 if k > 0 else -1
    return labels, {i: label for i, (label, _) in enumerate(themes)}
