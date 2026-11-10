"""Group complaint pains into clusters. Two methods (`clustering.method`, `cluster --method`):

  hdbscan  embeddings → PCA → HDBSCAN; each cluster is labeled by Claude from its 10 most
           central pains
  llm      Claude proposes ~15 themes from all pains, then assigns each pain (cluster_llm.py)

`auto` picks llm when a niche has fewer than `clustering.llm_max_pains` pains. Either way, groups
smaller than min_cluster_size become noise, and a warning is logged when noise exceeds
`clustering.noise_warning`. Each run writes new clusters (global ids) under one run_id;
config/merges.yaml is applied later by `score`."""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Sequence

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from . import db
from .cluster_llm import clean_label, llm_assign
from .extract import Completer
from .models import Cluster
from .settings import ClusterSettings
from .source_rules import strip_boilerplate

log = logging.getLogger(__name__)

__all__ = ["NOISE_LABEL", "assign_clusters", "choose_method", "clean_label", "cluster_niche",
           "core_text", "embed", "pick_representatives"]

NOISE_LABEL = "Ruido (sin agrupar)"
LABEL_SYSTEM = (
    "Recibes los dolores más representativos de un grupo de quejas escritas por personas en "
    "Colombia. Escribe una etiqueta en español de exactamente 6 palabras que nombre el problema "
    "común (el problema, no una solución). Responde solo con la etiqueta, sin comillas ni punto "
    "final."
)
_OPENER = re.compile(r"^(necesita|requiere|quiere|busca|desea)\s+(que\s+)?", re.I)
_models: dict[str, object] = {}


def core_text(pain: str) -> str:
    """Drop job-post boilerplate lines and the shared "Necesita…" opener so embeddings separate
    on the need itself."""
    pain = strip_boilerplate(pain)
    return _OPENER.sub("", pain).strip() or pain


def embed(texts: Sequence[str], model_name: str) -> np.ndarray:
    """L2-normalized sentence embeddings (so euclidean distance tracks cosine distance)."""
    from sentence_transformers import SentenceTransformer  # heavy import; only when clustering

    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name)
    model = _models[model_name]
    return np.asarray(model.encode(list(texts), normalize_embeddings=True,  # type: ignore[attr-defined]
                                   show_progress_bar=False, convert_to_numpy=True))


def choose_method(method: str, n_pains: int, cfg: ClusterSettings) -> str:
    if method == "auto":
        return "llm" if n_pains < cfg.llm_max_pains else "hdbscan"
    return method


def assign_clusters(embeddings: np.ndarray, cfg: ClusterSettings) -> np.ndarray:
    """HDBSCAN labels per row after optional PCA; -1 = noise."""
    if len(embeddings) < cfg.min_cluster_size:
        return np.full(len(embeddings), -1)
    points = embeddings
    if cfg.n_components and cfg.n_components < min(embeddings.shape):
        points = PCA(n_components=cfg.n_components, random_state=0).fit_transform(embeddings)
    model = HDBSCAN(min_cluster_size=cfg.min_cluster_size, min_samples=cfg.min_samples,
                    metric="euclidean", copy=True)
    return model.fit_predict(points)


def by_centrality(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(order, cosine distances to the normalized centroid) for one cluster's embeddings."""
    centroid = embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid) or 1.0
    distances = 1.0 - embeddings @ centroid
    return np.argsort(distances, kind="stable"), distances


def pick_representatives(rows: Sequence[sqlite3.Row], k: int) -> list[int]:
    """First k pains (rows already sorted by centrality) from distinct authors, taking scored
    (in-country) pains before ones excluded from scoring."""
    reps: list[int] = []
    authors: set[str | None] = set()
    for row in sorted(rows, key=lambda r: bool(r["exclude_from_scoring"])):  # stable sort
        author = row["author_hash"] or f"pain:{row['pain_id']}"
        if author in authors:
            continue
        reps.append(row["pain_id"])
        authors.add(author)
        if len(reps) == k:
            break
    return reps


def label_cluster(llm: Completer, model: str, pains: Sequence[str]) -> str:
    text = llm.complete(purpose="label", model=model, system=LABEL_SYSTEM,
                        user="\n".join(f"- {p}" for p in pains), n_items=1, max_tokens=4000)
    return clean_label(text)


def cluster_niche(conn: sqlite3.Connection, llm: Completer, niche: str, run_id: str,
                  cfg: ClusterSettings, label_model: str,
                  method: str | None = None) -> tuple[str, int, int]:
    """Cluster one niche's complaint pains. Returns (method used, clusters, pains in noise)."""
    rows = db.complaint_pains(conn, niche)
    chosen = choose_method(method or cfg.method, len(rows), cfg)
    if not rows:
        log.info("cluster: no complaint pains for %s yet", niche)
        return chosen, 0, 0
    embeddings = embed([core_text(r["pain"]) for r in rows], cfg.embedding_model)
    if chosen == "llm":
        labels, names = llm_assign(llm, label_model,
                                   [strip_boilerplate(r["pain"]) for r in rows], cfg)
    else:
        labels, names = assign_clusters(embeddings, cfg), {}
    for value in set(labels.tolist()) - {-1}:  # groups below min_cluster_size are noise
        if (labels == value).sum() < cfg.min_cluster_size:
            labels[labels == value] = -1
    groups = [(v, np.flatnonzero(labels == v)) for v in sorted(set(labels.tolist()) - {-1})]
    noise = np.flatnonzero(labels == -1)
    if len(noise):
        groups.append((-1, noise))
    for value, idx in groups:
        order, distances = by_centrality(embeddings[idx])
        ordered = [rows[idx[o]] for o in order]
        if value == -1:
            label = NOISE_LABEL
        else:
            label = names.get(value) or label_cluster(
                llm, label_model, [r["pain"] for r in ordered[: cfg.label_examples]])
        cluster = Cluster(label=label, niche_tag=niche, run_id=run_id, method=chosen,
                          is_noise=value == -1, size=len(idx),
                          representative_ids=pick_representatives(ordered, cfg.representatives))
        members = [(rows[idx[o]]["pain_id"], float(distances[o])) for o in order]
        cluster_id = db.insert_cluster(conn, cluster, members)
        log.info("cluster %d (%s): %d pains — %s", cluster_id, niche, len(idx), label)
    share = len(noise) / len(rows)
    if share > cfg.noise_warning:
        log.warning("cluster %s: %.0f%% of %d pains are noise (warning threshold %.0f%%); "
                    "consider another method or more data", niche, 100 * share, len(rows),
                    100 * cfg.noise_warning)
    return chosen, len(groups) - (1 if len(noise) else 0), len(noise)
