"""Collector registry. Each source reads only its own key from a niche entry in niches.yaml."""

from .base import Collector
from .computrabajo import ComputrabajoCollector
from .google_play import GooglePlayCollector
from .youtube import YouTubeCollector

REGISTRY: dict[str, type[Collector]] = {
    cls.source: cls for cls in (GooglePlayCollector, ComputrabajoCollector, YouTubeCollector)
}

__all__ = ["REGISTRY", "Collector"]
