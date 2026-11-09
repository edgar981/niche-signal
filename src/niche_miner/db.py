"""SQLite schema and query helpers for data/signals.db (plain sqlite3, rows as sqlite3.Row)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .models import Cluster, ExtractedPain, RawSignal, utcnow

if TYPE_CHECKING:
    from .score import ClusterStats

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_signals (
    id           INTEGER PRIMARY KEY,
    niche_tag    TEXT NOT NULL,
    source       TEXT NOT NULL,
    external_id  TEXT NOT NULL,
    url          TEXT,
    author_hash  TEXT,
    author_type  TEXT NOT NULL DEFAULT 'unknown'
                 CHECK (author_type IN ('business', 'consumer', 'unknown')),
    text         TEXT NOT NULL,
    lang         TEXT,
    posted_at    TEXT,
    collected_at TEXT NOT NULL,
    UNIQUE (source, external_id)
);
CREATE INDEX IF NOT EXISTS idx_raw_signals_niche ON raw_signals (niche_tag);

CREATE TABLE IF NOT EXISTS extracted_pains (
    id            INTEGER PRIMARY KEY,
    raw_signal_id INTEGER NOT NULL UNIQUE REFERENCES raw_signals (id) ON DELETE CASCADE,
    is_complaint  INTEGER,
    pain          TEXT,
    pain_type     TEXT,  -- missing_capability | incumbent_quality | pricing | onboarding
    who           TEXT,
    who_type      TEXT CHECK (who_type IN ('business', 'consumer', 'unknown')),
    country       TEXT,  -- co | mx | pe | ar | es | other | unknown
    exclude_from_scoring INTEGER NOT NULL DEFAULT 0,  -- 1: known non-Colombian, kept but not scored
    current_tool  TEXT,
    spend_signal  TEXT,
    severity      INTEGER CHECK (severity BETWEEN 1 AND 5),
    model         TEXT NOT NULL,
    extracted_at  TEXT NOT NULL,
    error         TEXT  -- non-null: extraction failed after one retry
);

CREATE TABLE IF NOT EXISTS clusters (
    id                 INTEGER PRIMARY KEY,
    label              TEXT NOT NULL,
    niche_tag          TEXT NOT NULL,
    run_id             TEXT NOT NULL,
    method             TEXT,  -- hdbscan | llm
    is_noise           INTEGER NOT NULL DEFAULT 0,
    merged_into        INTEGER REFERENCES clusters (id),
    size               INTEGER NOT NULL DEFAULT 0,  -- scored pains (after country exclusion)
    distinct_authors   INTEGER NOT NULL DEFAULT 0,
    spend_count        INTEGER NOT NULL DEFAULT 0,
    b2b_ratio          REAL NOT NULL DEFAULT 0,
    reachability       REAL NOT NULL DEFAULT 0,
    competition        REAL NOT NULL DEFAULT 0,
    score              REAL,
    score_breakdown    TEXT,                       -- JSON, written by `score`
    representative_ids TEXT NOT NULL DEFAULT '[]'  -- JSON list of extracted_pains.id
);
CREATE INDEX IF NOT EXISTS idx_clusters_run ON clusters (niche_tag, run_id);

CREATE TABLE IF NOT EXISTS cluster_members (
    cluster_id INTEGER NOT NULL REFERENCES clusters (id) ON DELETE CASCADE,
    pain_id    INTEGER NOT NULL REFERENCES extracted_pains (id) ON DELETE CASCADE,
    distance   REAL NOT NULL,  -- cosine distance to the cluster centroid
    PRIMARY KEY (cluster_id, pain_id)
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id            INTEGER PRIMARY KEY,
    run_id        TEXT NOT NULL,
    purpose       TEXT NOT NULL,  -- extract | label | themes | assign
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd      REAL,
    n_items       INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
"""

# Columns added after the first release; connect() adds any that an older DB lacks.
MIGRATIONS = [
    ("extracted_pains", "pain_type", "TEXT"),
    ("extracted_pains", "country", "TEXT"),
    ("extracted_pains", "exclude_from_scoring", "INTEGER NOT NULL DEFAULT 0"),
    ("clusters", "method", "TEXT"),
]


def connect(path: Path | str) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    for table, column, ddl in MIGRATIONS:
        if column not in {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    conn.commit()
    return conn


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


def _marks(n: int) -> str:
    return ",".join("?" * n)


def insert_signals(conn: sqlite3.Connection, signals: Iterable[RawSignal]) -> int:
    """Insert signals, skipping (source, external_id) duplicates. Returns the number of new rows."""
    rows = [(s.niche_tag, s.source, s.external_id, s.url, s.author_hash, s.author_type,
             s.text, s.lang, _iso(s.posted_at), _iso(s.collected_at)) for s in signals]
    before = conn.total_changes
    with conn:
        conn.executemany(
            """INSERT OR IGNORE INTO raw_signals (niche_tag, source, external_id, url, author_hash,
               author_type, text, lang, posted_at, collected_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            rows)
    return conn.total_changes - before


def external_ids(conn: sqlite3.Connection, source: str) -> set[str]:
    rows = conn.execute("SELECT external_id FROM raw_signals WHERE source = ?", (source,))
    return {r[0] for r in rows}


def pending_signals(conn: sqlite3.Connection, niche: str | None = None,
                    limit: int | None = None) -> list[sqlite3.Row]:
    """Raw signals without an extraction row yet, oldest first."""
    sql = """SELECT rs.* FROM raw_signals rs
             LEFT JOIN extracted_pains ep ON ep.raw_signal_id = rs.id
             WHERE ep.id IS NULL AND (?1 IS NULL OR rs.niche_tag = ?1) ORDER BY rs.id"""
    params: list[object] = [niche]
    if limit:
        sql += " LIMIT ?2"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def save_pains(conn: sqlite3.Connection,
               results: Sequence[tuple[int, ExtractedPain | None, str | None]], model: str) -> None:
    """Store (raw_signal_id, pain or None, error) triples; a None pain is a failed extraction."""
    now = _iso(utcnow())
    rows = []
    for signal_id, p, error in results:
        if p is None:
            rows.append((signal_id, None, None, None, None, None, None, 0, None, None, None,
                         model, now, error or "unknown error"))
        else:
            rows.append((signal_id, int(p.is_complaint), p.pain, p.pain_type, p.who, p.who_type,
                         p.country, int(p.excluded_from_scoring), p.current_tool, p.spend_signal,
                         p.severity, model, now, None))
    with conn:
        conn.executemany(
            """INSERT INTO extracted_pains (raw_signal_id, is_complaint, pain, pain_type, who,
               who_type, country, exclude_from_scoring, current_tool, spend_signal, severity,
               model, extracted_at, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)


def delete_extractions(conn: sqlite3.Connection, niche: str | None = None,
                       failed_only: bool = False) -> int:
    """Delete extraction rows (cluster memberships go with them) so signals are re-extracted."""
    with conn:
        cur = conn.execute(
            """DELETE FROM extracted_pains WHERE (?2 = 0 OR error IS NOT NULL) AND raw_signal_id IN
               (SELECT id FROM raw_signals WHERE ?1 IS NULL OR niche_tag = ?1)""",
            (niche, int(failed_only)))
    return cur.rowcount


def complaint_pains(conn: sqlite3.Connection, niche: str) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT ep.id AS pain_id, ep.pain, ep.exclude_from_scoring, rs.author_hash
           FROM extracted_pains ep JOIN raw_signals rs ON rs.id = ep.raw_signal_id
           WHERE rs.niche_tag = ? AND ep.is_complaint = 1 AND ep.error IS NULL
             AND ep.pain IS NOT NULL
           ORDER BY ep.id""", (niche,)).fetchall()


def niches_with_complaints(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """SELECT DISTINCT rs.niche_tag FROM extracted_pains ep
           JOIN raw_signals rs ON rs.id = ep.raw_signal_id
           WHERE ep.is_complaint = 1 ORDER BY 1""").fetchall()
    return [r[0] for r in rows]


def insert_cluster(conn: sqlite3.Connection, cluster: Cluster,
                   members: Sequence[tuple[int, float]]) -> int:
    """Insert a cluster and its (pain_id, distance) members; returns the new cluster id."""
    with conn:
        cur = conn.execute(
            """INSERT INTO clusters (label, niche_tag, run_id, method, is_noise, size,
               representative_ids) VALUES (?,?,?,?,?,?,?)""",
            (cluster.label, cluster.niche_tag, cluster.run_id, cluster.method,
             int(cluster.is_noise), cluster.size, json.dumps(cluster.representative_ids)))
        cluster_id = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO cluster_members (cluster_id, pain_id, distance) VALUES (?,?,?)",
            [(cluster_id, pain_id, distance) for pain_id, distance in members])
    return cluster_id


def latest_clusters(conn: sqlite3.Connection, niche: str | None = None) -> list[sqlite3.Row]:
    """All clusters (noise included) from the most recent run of each niche."""
    return conn.execute(
        """SELECT c.* FROM clusters c
           JOIN (SELECT niche_tag, MAX(run_id) AS run_id FROM clusters GROUP BY niche_tag) last
             ON last.niche_tag = c.niche_tag AND last.run_id = c.run_id
           WHERE ?1 IS NULL OR c.niche_tag = ?1
           ORDER BY c.id""", (niche,)).fetchall()


def member_details(conn: sqlite3.Connection, cluster_ids: Sequence[int]) -> list[sqlite3.Row]:
    """Members of one cluster or merge group with pain + source fields, most central first."""
    return conn.execute(
        f"""SELECT cm.cluster_id, cm.distance, ep.id AS pain_id, ep.pain, ep.pain_type, ep.who,
                   ep.who_type, ep.country, ep.exclude_from_scoring, ep.current_tool,
                   ep.spend_signal, ep.severity, rs.id AS signal_id, rs.source, rs.url, rs.text,
                   rs.author_hash, rs.author_type
            FROM cluster_members cm
            JOIN extracted_pains ep ON ep.id = cm.pain_id
            JOIN raw_signals rs ON rs.id = ep.raw_signal_id
            WHERE cm.cluster_id IN ({_marks(len(cluster_ids))})
            ORDER BY cm.distance, ep.id""", list(cluster_ids)).fetchall()


def save_score(conn: sqlite3.Connection, cluster_id: int, stats: ClusterStats, *,
               reachability: float = 0, competition: float = 0, score: float | None = None,
               breakdown: dict | None = None, merged_into: int | None = None) -> None:
    conn.execute(
        """UPDATE clusters SET size=?, distinct_authors=?, spend_count=?, b2b_ratio=?,
           reachability=?, competition=?, score=?, score_breakdown=?, merged_into=?
           WHERE id=?""",
        (stats.size, stats.distinct_authors, stats.spend_count,
         stats.business_count / stats.size if stats.size else 0.0, reachability, competition,
         score, json.dumps(breakdown) if breakdown else None, merged_into, cluster_id))


def log_llm_call(conn: sqlite3.Connection, run_id: str, purpose: str, model: str,
                 input_tokens: int, output_tokens: int, cost_usd: float | None,
                 n_items: int) -> None:
    with conn:
        conn.execute(
            """INSERT INTO llm_calls (run_id, purpose, model, input_tokens, output_tokens,
               cost_usd, n_items, created_at) VALUES (?,?,?,?,?,?,?,?)""",
            (run_id, purpose, model, input_tokens, output_tokens, cost_usd, n_items,
             _iso(utcnow())))


def llm_usage(conn: sqlite3.Connection, run_id: str | None = None) -> list[sqlite3.Row]:
    """Calls, tokens and cost per purpose — for one run, or all time when run_id is None."""
    return conn.execute(
        """SELECT purpose, COUNT(*) AS calls, SUM(input_tokens) AS input_tokens,
                  SUM(output_tokens) AS output_tokens, COALESCE(SUM(cost_usd), 0) AS cost_usd,
                  SUM(cost_usd IS NULL) AS unpriced, GROUP_CONCAT(DISTINCT model) AS models,
                  SUM(n_items) AS items, MAX(run_id) AS last_run
           FROM llm_calls WHERE ?1 IS NULL OR run_id = ?1
           GROUP BY purpose ORDER BY purpose""", (run_id,)).fetchall()


def signal_counts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT niche_tag, source, COUNT(*) AS n FROM raw_signals
           GROUP BY niche_tag, source ORDER BY niche_tag, source""").fetchall()


def extraction_counts(conn: sqlite3.Connection) -> sqlite3.Row:
    return conn.execute(
        """SELECT COUNT(*) AS total, COALESCE(SUM(error IS NOT NULL), 0) AS failed,
                  COALESCE(SUM(is_complaint = 1), 0) AS complaints,
                  COALESCE(SUM(is_complaint = 0), 0) AS not_complaints,
                  COALESCE(SUM(is_complaint = 1 AND exclude_from_scoring = 1), 0) AS excluded
           FROM extracted_pains""").fetchone()


def complaint_breakdown(conn: sqlite3.Connection, column: str) -> list[sqlite3.Row]:
    """Complaint counts by pain_type or country (most common first)."""
    if column not in ("pain_type", "country"):
        raise ValueError(column)
    return conn.execute(
        f"""SELECT COALESCE({column}, 'unknown') AS value, COUNT(*) AS n FROM extracted_pains
            WHERE is_complaint = 1 GROUP BY 1 ORDER BY 2 DESC, 1""").fetchall()
