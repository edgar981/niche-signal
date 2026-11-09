"""Shared text filters: minimum length, obvious spam, in-run duplicates, Spanish only (langdetect)."""

from __future__ import annotations

import re
from collections import Counter

from langdetect import DetectorFactory, LangDetectException, detect

from ..models import RawSignal
from ..settings import FilterSettings

DetectorFactory.seed = 0  # langdetect is randomized by default; make runs reproducible

_URL = re.compile(r"https?://|www\.", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?57[\s-]?)?3\d{2}[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")  # CO mobile
_PROMO = re.compile(
    r"gana(?:r)? dinero|ingresos extra|desde casa|sin experiencia|inversi[oó]n garantizada|"
    r"escr[ií]beme|cont[aá]ctame|link en (?:mi )?perfil|env[ií]a(?:me)? un mensaje|"
    r"criptomoneda|forex|sorteo|s[ií]gueme",
    re.I,
)


def detect_language(text: str) -> str | None:
    try:
        return detect(text)
    except LangDetectException:  # no detectable features (emoji-only, numbers...)
        return None


def is_spam(text: str) -> bool:
    """Conservative: only drops texts that are clearly ads or junk."""
    letters = sum(ch.isalpha() for ch in text)
    if letters < 0.5 * len(text):  # mostly emoji, digits or symbols
        return True
    if (_URL.search(text) or _PHONE.search(text)) and _PROMO.search(text):
        return True
    if len(_URL.findall(text)) >= 2:
        return True
    words = text.casefold().split()
    return len(words) >= 8 and len(set(words)) / len(words) < 0.3  # copy-paste repetition


def _dedupe_key(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


class TextFilter:
    """Stateful per collector run: also drops exact (normalized) duplicate texts."""

    def __init__(self, cfg: FilterSettings):
        self.cfg = cfg
        self.dropped: Counter[str] = Counter()
        self._seen: set[str] = set()

    def reason_to_drop(self, text: str) -> str | None:
        if len(text) < self.cfg.min_chars:
            return "short"
        if is_spam(text):
            return "spam"
        if _dedupe_key(text) in self._seen:
            return "duplicate"
        if detect_language(text) not in self.cfg.languages:
            return "language"
        return None

    def keep(self, signal: RawSignal) -> bool:
        """True if the signal passes; sets signal.lang. Records why others were dropped."""
        signal.text = signal.text.strip()
        reason = self.reason_to_drop(signal.text)
        if reason:
            self.dropped[reason] += 1
            return False
        self._seen.add(_dedupe_key(signal.text))
        signal.lang = detect_language(signal.text)
        return True
