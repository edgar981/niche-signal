"""HDBSCAN and LLM-theme clustering with synthetic embeddings (no model download, fake LLM)."""

import json
import logging

import numpy as np
import pytest
from helpers import FakeLLM, seed_pain

from niche_miner import cluster as clustering
from niche_miner import db
from niche_miner.settings import ClusterSettings


def blobs(per_group: int = 8, groups: int = 3, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(groups, dim)) * 5
    points = np.vstack([c + rng.normal(scale=0.05, size=(per_group, dim)) for c in centers])
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def fake_embeddings(monkeypatch, embeddings: np.ndarray) -> None:
    monkeypatch.setattr(clustering, "embed", lambda texts, model: embeddings[: len(texts)])


def test_hdbscan_finds_separated_groups_after_pca():
    labels = clustering.assign_clusters(blobs(), ClusterSettings(n_components=10, min_samples=2))
    assert len(set(labels.tolist()) - {-1}) == 3


def test_fewer_points_than_min_cluster_size_are_all_noise():
    labels = clustering.assign_clusters(blobs(per_group=1, groups=3), ClusterSettings())
    assert set(labels.tolist()) == {-1}


@pytest.mark.parametrize(("method", "pains", "chosen"), [
    ("auto", 1499, "llm"), ("auto", 1500, "hdbscan"), ("hdbscan", 10, "hdbscan"),
    ("llm", 5000, "llm")])
def test_method_choice(method, pains, chosen):
    assert clustering.choose_method(method, pains, ClusterSettings(llm_max_pains=1500)) == chosen


def test_shared_opener_is_stripped_before_embedding():
    assert clustering.core_text("Necesita que la app no se cierre.") == "la app no se cierre."
    assert clustering.core_text("Requiere conciliar bancos") == "conciliar bancos"


def rep_row(pain_id: int, author: str | None, excluded: int = 0) -> dict:
    return {"pain_id": pain_id, "author_hash": author, "exclude_from_scoring": excluded}


def test_representatives_come_from_distinct_authors():
    rows = [rep_row(1, "a"), rep_row(2, "a"), rep_row(3, "b"), rep_row(4, None), rep_row(5, "c")]
    assert clustering.pick_representatives(rows, 3) == [1, 3, 4]


def test_representatives_prefer_pains_that_are_scored():
    rows = [rep_row(1, "mx", excluded=1), rep_row(2, "a"), rep_row(3, "b"), rep_row(4, "c")]
    assert clustering.pick_representatives(rows, 3) == [2, 3, 4]
    assert clustering.pick_representatives(rows[:2], 3) == [2, 1]  # excluded ones fill last


@pytest.mark.parametrize(("raw", "label"), [
    ("Facturación electrónica rechazada por la DIAN", "Facturación electrónica rechazada por la DIAN"),
    ('"Etiqueta: Soporte técnico lento y sin respuesta."\n', "Soporte técnico lento y sin respuesta"),
    ("**Cobros dobles sin reembolso del dinero**", "Cobros dobles sin reembolso del dinero"),
    ("", "Sin etiqueta"),
])
def test_clean_label(raw, label):
    assert clustering.clean_label(raw) == label


def test_hdbscan_method_stores_clusters_members_and_noise(conn, monkeypatch):
    rng = np.random.default_rng(1)
    outliers = rng.normal(size=(2, 16))
    outliers /= np.linalg.norm(outliers, axis=1, keepdims=True)
    embeddings = np.vstack([blobs(per_group=6, groups=2), outliers])
    for i in range(len(embeddings)):
        seed_pain(conn, i, author=f"author-{i}")
    fake_embeddings(monkeypatch, embeddings)
    llm = FakeLLM("Primer grupo de quejas muy parecidas", "Segundo grupo de quejas muy parecidas")

    used, created, noise = clustering.cluster_niche(conn, llm, "test_niche", "run-1",
                                                    ClusterSettings(), "fake-model", "hdbscan")

    assert (used, created, noise) == ("hdbscan", 2, 2)
    rows = db.latest_clusters(conn)
    assert sorted(r["label"] for r in rows) == ["Primer grupo de quejas muy parecidas",
                                                "Ruido (sin agrupar)",
                                                "Segundo grupo de quejas muy parecidas"]
    assert {r["method"] for r in rows} == {"hdbscan"}
    assert conn.execute("SELECT COUNT(*) FROM cluster_members").fetchone()[0] == 14
    assert all(len(call["user"].splitlines()) <= 10 for call in llm.calls)


def themes_reply(*labels: str) -> str:
    return json.dumps({"themes": [{"label": label, "description": f"Dolores sobre {label}"}
                                  for label in labels]})


def assign_reply(pairs: dict[int, int]) -> str:
    return json.dumps({"assignments": [{"id": j, "theme": k} for j, k in pairs.items()]})


def test_llm_method_groups_pains_by_proposed_themes(conn, monkeypatch, caplog):
    for i in range(14):
        seed_pain(conn, i, author=f"author-{i}")
    fake_embeddings(monkeypatch, blobs(per_group=7, groups=2))
    cfg = ClusterSettings(llm_batch_size=8)
    llm = FakeLLM(
        themes_reply("Facturación desde el celular", "Cobros dobles sin reembolso",
                     "Tema con muy pocos dolores"),
        assign_reply({j: 1 for j in range(6)}),        # batch 0-7: ids 6 and 7 missing
        assign_reply({0: 3, 1: 0}),                     # retry: 6 → small theme, 7 → none
        assign_reply({j: 2 for j in range(6)}),        # batch 8-13: all theme 2
    )

    with caplog.at_level(logging.WARNING):
        used, created, noise = clustering.cluster_niche(conn, llm, "test_niche", "run-1", cfg,
                                                        "fake-model")

    assert (used, created, noise) == ("llm", 2, 2)
    rows = {r["label"]: r for r in db.latest_clusters(conn)}
    assert set(rows) == {"Facturación desde el celular", "Cobros dobles sin reembolso",
                         "Ruido (sin agrupar)"}
    assert rows["Facturación desde el celular"]["size"] == 6
    assert {r["method"] for r in rows.values()} == {"llm"}
    assert [c["purpose"] for c in llm.calls] == ["themes", "assign", "assign", "assign"]
    assert llm.calls[0]["effort"] == "medium"
    assert "noise" not in caplog.text  # 2 of 14 is below the warning threshold


def test_noise_above_threshold_logs_a_warning(conn, monkeypatch, caplog):
    for i in range(10):
        seed_pain(conn, i, author=f"author-{i}")
    fake_embeddings(monkeypatch, blobs(per_group=5, groups=2))
    llm = FakeLLM(themes_reply("Facturación desde el celular"),
                  assign_reply({j: (1 if j < 2 else 0) for j in range(10)}))

    with caplog.at_level(logging.WARNING):
        used, created, noise = clustering.cluster_niche(conn, llm, "test_niche", "run-1",
                                                        ClusterSettings(), "fake-model", "llm")

    assert (created, noise) == (0, 10)  # the one theme got 2 pains, below min_cluster_size
    assert "100% of 10 pains are noise" in caplog.text
