import sqlite3

import pytest
from helpers import make_signal

from niche_miner import db


def test_reinserting_signals_creates_no_duplicates(conn):
    signals = [make_signal(i) for i in range(5)]
    assert db.insert_signals(conn, signals) == 5
    assert db.insert_signals(conn, signals) == 0
    # Same (source, external_id) seen again in a later run with edited text: still a duplicate.
    edited = make_signal(0, text="El mismo review editado por el autor después de una semana.")
    assert db.insert_signals(conn, [edited]) == 0
    assert conn.execute("SELECT COUNT(*) FROM raw_signals").fetchone()[0] == 5


def test_dedupe_key_includes_the_source(conn):
    pair = [make_signal(0, source="google_play"), make_signal(0, source="youtube")]
    assert db.insert_signals(conn, pair) == 2


def test_author_type_is_constrained(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("""INSERT INTO raw_signals (niche_tag, source, external_id, text,
                        collected_at, author_type) VALUES ('n', 's', 'e', 't', 'now', 'robot')""")


def test_a_first_release_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE extracted_pains (id INTEGER PRIMARY KEY, raw_signal_id INTEGER NOT NULL UNIQUE,
            is_complaint INTEGER, pain TEXT, who TEXT, who_type TEXT, current_tool TEXT,
            spend_signal TEXT, severity INTEGER, model TEXT NOT NULL, extracted_at TEXT NOT NULL,
            error TEXT);
        INSERT INTO extracted_pains (raw_signal_id, is_complaint, pain, model, extracted_at)
            VALUES (1, 1, 'Necesita algo', 'old-model', '2026-11-08');
        CREATE TABLE clusters (id INTEGER PRIMARY KEY, label TEXT NOT NULL, niche_tag TEXT NOT NULL,
            run_id TEXT NOT NULL);""")
    old.close()
    conn = db.connect(path)
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(extracted_pains)")}
    assert {"pain_type", "country", "exclude_from_scoring", "cost_of_problem"} <= columns
    assert "method" in {r["name"] for r in conn.execute("PRAGMA table_info(clusters)")}
    row = conn.execute("SELECT pain, exclude_from_scoring FROM extracted_pains").fetchone()
    assert tuple(row) == ("Necesita algo", 0)  # existing rows kept, flag defaults to in scope
    conn.close()


def test_pending_signals_skips_extracted_and_filters_by_niche(conn):
    db.insert_signals(conn, [make_signal(0, niche="a"), make_signal(1, niche="a"),
                             make_signal(2, niche="b")])
    first = db.pending_signals(conn, "a")[0]
    db.save_pains(conn, [(first["id"], None, "boom")], "fake-model")
    assert [r["external_id"] for r in db.pending_signals(conn, "a")] == ["ext-1"]
    assert len(db.pending_signals(conn)) == 2
    assert len(db.pending_signals(conn, limit=1)) == 1
    assert db.delete_extractions(conn, "a", failed_only=True) == 1
    assert len(db.pending_signals(conn, "a")) == 2
