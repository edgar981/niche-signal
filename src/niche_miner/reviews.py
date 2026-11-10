"""Review inputs for scoring: human overrides (config/reachability.yaml), merges
(config/merges.yaml) and the automated review cached by `niche-miner review`:
competition per cluster (data/review/<cluster_id>.json) and reachability per niche
(data/review/niche_<tag>.json).

config/reachability.yaml holds overrides only:

  clusters:
    64: {reachability: null, competition: 7, status: active, note: ""}

  reachability / competition  0-10, or null to use the automated value
  status                      active | not_a_niche | merged_into:<id>
  note                        free text; a note alone marks the cluster as reviewed

Ids are global (each `cluster` run creates new ones); entries for ids that aren't in the latest
run are reported and ignored."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .settings import load_yaml

STATUSES = ("active", "not_a_niche", "merged_into:<id>")
_MERGED = re.compile(r"^merged_into\s*:\s*(\d+)$")


@dataclass(frozen=True)
class Override:
    reachability: float | None = None
    competition: float | None = None
    status: str = "active"  # active | not_a_niche | merged
    merged_into: int | None = None
    note: str = ""

    @property
    def reviewed(self) -> bool:
        """A person looked at it: a number, a status other than active, or a note."""
        return (self.status != "active" or self.reachability is not None
                or self.competition is not None or bool(self.note))


def _number(value: Any, field: str, cid: int, warnings: list[str]) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        warnings.append(f"reachability.yaml: {cid}.{field} = {value!r} is not a number; ignored")
        return None
    if not 0 <= value <= 10:
        warnings.append(f"reachability.yaml: {cid}.{field} = {value} is outside 0-10; clamped")
    return float(max(0, min(10, value)))


def _status(value: Any, cid: int, warnings: list[str]) -> tuple[str, int | None]:
    if isinstance(value, dict) and "merged_into" in value:  # {merged_into: 64}
        value = f"merged_into:{value['merged_into']}"
    text = str(value or "active").strip()
    if text in ("active", "not_a_niche"):
        return text, None
    if m := _MERGED.match(text):
        return "merged", int(m.group(1))
    warnings.append(f"reachability.yaml: {cid}.status = {text!r} is not one of "
                    f"{', '.join(STATUSES)}; treated as active")
    return "active", None


def load_overrides(path: Path, current: set[int] | None = None
                   ) -> tuple[dict[int, Override], dict[int, list[int]], list[str]]:
    """(overrides by cluster id, merges implied by merged_into statuses, warnings)."""
    warnings: list[str] = []
    overrides: dict[int, Override] = {}
    merges: dict[int, list[int]] = {}
    for key, entry in (load_yaml(path).get("clusters") or {}).items():
        cid = int(key)
        if current is not None and cid not in current:
            warnings.append(f"reachability.yaml: {cid} is not in the latest run; ignored")
            continue
        entry = entry if isinstance(entry, dict) else {}
        status, target = _status(entry.get("status"), cid, warnings)
        overrides[cid] = Override(
            reachability=_number(entry.get("reachability"), "reachability", cid, warnings),
            competition=_number(entry.get("competition"), "competition", cid, warnings),
            status=status, merged_into=target, note=str(entry.get("note") or "").strip())
        if target is not None:
            merges.setdefault(target, []).append(cid)
    return overrides, merges, warnings


def load_merges(path: Path) -> dict[int, list[int]]:
    raw = load_yaml(path).get("merges") or {}
    return {int(target): [int(s) for s in (sources or [])] for target, sources in raw.items()}


def resolve_merges(cluster_ids: set[int],
                   merges: dict[int, list[int]]) -> tuple[dict[int, int], list[str]]:
    """Map every current cluster id to the id of the group it ends up in (chains allowed).
    Ids that aren't in the latest run are reported and ignored."""
    owner = {cid: cid for cid in cluster_ids}
    warnings: list[str] = []
    for target, sources in merges.items():
        if target not in owner:
            warnings.append(f"merge target {target} is not in the latest run; skipped")
            continue
        for source in sources:
            if source not in owner:
                warnings.append(f"merge source {source} is not in the latest run; skipped")
                continue
            old, new = owner[source], owner[target]
            if old != new:
                for cid, own in owner.items():
                    if own == old:
                        owner[cid] = new
    return owner, warnings


def review_path(review_dir: Path, cluster_id: int) -> Path:
    return review_dir / f"{cluster_id}.json"


def load_auto(review_dir: Path | None, cluster_id: int) -> dict[str, Any]:
    """The cached automated review for one cluster, or {} if there is none."""
    if review_dir is None or not (path := review_path(review_dir, cluster_id)).is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def niche_review_path(review_dir: Path, niche: str) -> Path:
    return review_dir / f"niche_{niche}.json"


def load_niche_auto(review_dir: Path | None, niche: str) -> dict[str, Any]:
    """The cached niche reachability review, or {} if there is none."""
    if review_dir is None or not (path := niche_review_path(review_dir, niche)).is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}
