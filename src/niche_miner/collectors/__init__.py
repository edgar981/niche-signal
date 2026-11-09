"""Collector registry. Each source reads only its own key from a niche entry in niches.yaml."""

from .base import Collector
from .computrabajo import ComputrabajoCollector
from .google_play import GooglePlayCollector

REGISTRY: dict[str, type[Collector]] = {
    cls.source: cls for cls in (GooglePlayCollector, ComputrabajoCollector)
}

__all__ = ["REGISTRY", "Collector"]
