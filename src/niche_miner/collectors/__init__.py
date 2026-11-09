"""Collector registry. Each source reads only its own key from a niche entry in niches.yaml."""

from .base import Collector
from .google_play import GooglePlayCollector

REGISTRY: dict[str, type[Collector]] = {
    cls.source: cls for cls in (GooglePlayCollector,)
}

__all__ = ["REGISTRY", "Collector"]
