"""`niche-miner calibrate`: set scoring.frequency_ceiling to the 90th percentile of the
cluster frequency sums (Σ_document min(ln(1 + authors), ln(1 + frequency_doc_cap))) over the
latest run of every niche collected, and stamp scoring.frequency_recalibrated_at.

Meant to run once, after the 35-niche Computrabajo pass; after that the ceiling stays fixed,
so a second run needs --force. Merged clusters count once, with their members; noise and
clusters left with no Colombian pains are skipped."""

from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from . import db
from .score import compute_stats, resolve_groups
from .settings import ScoreWeights

PERCENTILE = 90
_KEYS = ("frequency_ceiling", "frequency_recalibrated_at")


class CalibrationError(Exception):
    pass


@dataclass(frozen=True)
class Calibration:
    ceiling: float
    clusters: int
    niches: int
    previous: float
    sums: tuple[float, ...]


def percentile(values: Sequence[float], p: float) -> float:
    """Linear interpolation between closest ranks (numpy's default method)."""
    if not values:
        raise CalibrationError("no clusters to calibrate on")
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def cluster_sums(conn: sqlite3.Connection, config_dir: Path, w: ScoreWeights
                 ) -> tuple[list[float], int]:
    """(frequency sum per scored cluster, number of niches they come from)."""
    clusters, _, owner, groups = resolve_groups(conn, config_dir)
    sums, niches = [], set()
    for c in clusters:
        if c["is_noise"] or owner[c["id"]] != c["id"]:
            continue
        stats = compute_stats(db.member_details(conn, sorted(groups[c["id"]])), w)
        if stats.size:
            sums.append(stats.url_spread)
            niches.add(c["niche_tag"])
    return sums, len(niches)


def calibrate(conn: sqlite3.Connection, config_dir: Path, w: ScoreWeights,
              force: bool = False) -> Calibration:
    if w.frequency_recalibrated_at and not force:
        raise CalibrationError(
            f"the frequency ceiling was calibrated on {w.frequency_recalibrated_at} and stays "
            "fixed; pass --force to recalibrate")
    sums, niches = cluster_sums(conn, config_dir, w)
    ceiling = round(percentile(sums, PERCENTILE), 2)
    if ceiling <= 0:
        raise CalibrationError("every cluster sum is 0; nothing to calibrate on")
    return Calibration(ceiling, len(sums), niches, w.frequency_ceiling, tuple(sums))


def write_ceiling(settings_path: Path, ceiling: float, when: date) -> None:
    """Rewrite the two scoring lines in place, keeping every comment in settings.yaml."""
    text = settings_path.read_text(encoding="utf-8")
    values = {"frequency_ceiling": f"{ceiling:g}", "frequency_recalibrated_at": when.isoformat()}
    for key in _KEYS:
        pattern = re.compile(rf"^(\s+{key}:[ \t]*)[^\s#]*", re.M)
        if len(pattern.findall(text)) != 1:
            raise CalibrationError(f"{settings_path} needs exactly one `{key}:` line under "
                                   "`scoring:`")
        text = pattern.sub(lambda m, v=values[key]: m.group(1) + v, text)
    settings_path.write_text(text, encoding="utf-8")
