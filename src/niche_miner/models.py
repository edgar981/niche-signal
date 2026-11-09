"""Pydantic schemas shared across the pipeline."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AuthorType = Literal["business", "consumer", "unknown"]
Country = Literal["co", "mx", "pe", "ar", "es", "other", "unknown"]
PainType = Literal["missing_capability", "incumbent_quality", "pricing", "onboarding"]
# Pains from a known non-Colombian country are kept in the DB but not scored.
SCORED_COUNTRIES = frozenset({"co", "unknown"})

# Strings the LLM sometimes uses instead of a real null.
_NULLISH = {"", "null", "none", "n/a", "na", "-", "ninguno", "ninguna", "desconocido", "no aplica"}
_GENERIC_WHO = re.compile(r"^(?:el |la |un |una )?usuari[oa]s?\b", re.I)
_PACKAGE_ID = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+")


def utcnow() -> datetime:
    return datetime.now(UTC)


class RawSignal(BaseModel):
    """One public text as collected. author_hash is SHA-256(source:author id) — never a name."""

    niche_tag: str
    source: str
    external_id: str
    url: str | None = None
    author_hash: str | None = None
    author_type: AuthorType = "unknown"
    text: str
    lang: str | None = None
    posted_at: datetime | None = None
    collected_at: datetime = Field(default_factory=utcnow)


class ExtractedPain(BaseModel):
    """One validated LLM extraction result for a raw signal. The validators also enforce two
    prompt rules the model can break: no generic "usuario de …" roles, no package ids as tools."""

    model_config = ConfigDict(extra="ignore")

    is_complaint: bool
    pain: str | None = None
    pain_type: PainType | None = None
    who: str | None = None
    who_type: AuthorType = "unknown"
    country: Country = "unknown"
    current_tool: str | None = None
    spend_signal: str | None = None
    severity: int = Field(ge=1, le=5)

    @field_validator("pain", "who", "current_tool", "spend_signal", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return None if value.casefold() in _NULLISH else value
        return value

    @field_validator("who")
    @classmethod
    def _concrete_role(cls, value: str | None) -> str | None:
        return None if value and _GENERIC_WHO.match(value) else value

    @field_validator("current_tool")
    @classmethod
    def _product_name(cls, value: str | None) -> str | None:
        return None if value and _PACKAGE_ID.fullmatch(value) else value

    @model_validator(mode="after")
    def _complaint_fields(self) -> ExtractedPain:
        if not self.is_complaint:
            self.pain_type = None
            return self
        if not self.pain:
            raise ValueError("is_complaint is true but pain is empty")
        if not self.pain_type:
            raise ValueError("is_complaint is true but pain_type is missing")
        return self

    @property
    def excluded_from_scoring(self) -> bool:
        return self.country not in SCORED_COUNTRIES


class Cluster(BaseModel):
    id: int | None = None
    label: str
    niche_tag: str
    run_id: str
    method: str = "hdbscan"
    is_noise: bool = False
    size: int = 0
    distinct_authors: int = 0
    spend_count: int = 0
    b2b_ratio: float = 0.0
    reachability: float = 0.0
    competition: float = 0.0
    score: float | None = None
    representative_ids: list[int] = Field(default_factory=list)
