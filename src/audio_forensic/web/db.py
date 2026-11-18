"""SQLite + files. Every track (original, isolated, residual) is a list of WAV blocks."""

import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from ..backends import Cancelled, check
from ..config import DATA_DIR, SR
from ..media import peaks, read_wav, write_wav

DB_PATH = DATA_DIR / "app.sqlite3"
SOURCES_DIR = DATA_DIR / "sources"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL,                -- audio | video | youtube
    name       TEXT NOT NULL,
    path       TEXT,                         -- uploaded file, or saved YouTube copy
    url        TEXT,                         -- YouTube link
    yt_mode    TEXT,                         -- stream | save
    offset_s   REAL NOT NULL DEFAULT 0,      -- YouTube trim start
    duration_s REAL NOT NULL DEFAULT 0,
    has_video  INTEGER NOT NULL DEFAULT 0,
    splits     TEXT NOT NULL DEFAULT '[]',   -- timeline split points (s)
    error      TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS tracks (
    id        INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    job_id    INTEGER,
    role      TEXT NOT NULL,                 -- original | target | residual
    name      TEXT NOT NULL,
    color     TEXT,
    muted     INTEGER NOT NULL DEFAULT 0,
    solo      INTEGER NOT NULL DEFAULT 0,
    gain      REAL NOT NULL DEFAULT 100,     -- %
    edits     TEXT NOT NULL DEFAULT '[]',    -- [{a, b, gain, muted}] per clip
    status    TEXT NOT NULL DEFAULT 'partial', -- partial | complete | failed | cancelled
    covered   INTEGER NOT NULL DEFAULT 0     -- samples available from 0
);
CREATE TABLE IF NOT EXISTS segments (
    id       INTEGER PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    start    INTEGER NOT NULL,               -- sample offset
    n        INTEGER NOT NULL,               -- samples
    path     TEXT NOT NULL,
    peaks    BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id             INTEGER PRIMARY KEY,
    source_id      INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    input_track_id INTEGER NOT NULL,
    model          TEXT NOT NULL,
    device         TEXT NOT NULL,
    prompt         TEXT NOT NULL,
    anchors        TEXT NOT NULL DEFAULT '[]',
    visual_path    TEXT,
    range_start    REAL,
    range_end      REAL,
    options        TEXT NOT NULL DEFAULT '{}',
    status         TEXT NOT NULL DEFAULT 'queued', -- queued | running | done | failed | cancelled
    progress       REAL NOT NULL DEFAULT 0,
    message        TEXT,
    elapsed_s      REAL,
    peak_vram_mib  REAL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS seg_track ON segments(track_id, start);
"""
JSON_COLS = {"splits", "edits", "anchors", "options"}
_local = threading.local()


def conn() -> sqlite3.Connection:
    if getattr(_local, "conn", None) is None:
        c = sqlite3.connect(DB_PATH, isolation_level=None, check_same_thread=False, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        c.execute("PRAGMA journal_mode = WAL")
        _local.conn = c
    return _local.conn


def _row(r: sqlite3.Row | None) -> dict | None:
    return None if r is None else {k: json.loads(r[k]) if k in JSON_COLS and r[k] else r[k] for k in r.keys()}


def q(sql: str, *args) -> list[dict]:
    return [_row(r) for r in conn().execute(sql, args)]


def one(sql: str, *args) -> dict | None:
    return _row(conn().execute(sql, args).fetchone())


def ex(sql: str, *args) -> int:
    return conn().execute(sql, args).lastrowid


def insert(table: str, **cols) -> int:
    cols = {k: json.dumps(v) if k in JSON_COLS else v for k, v in cols.items()}
    return ex(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", *cols.values())


def update(table: str, id_: int, **cols) -> None:
    if cols:
        cols = {k: json.dumps(v) if k in JSON_COLS else v for k, v in cols.items()}
        ex(f"UPDATE {table} SET {', '.join(f'{k} = ?' for k in cols)} WHERE id = ?", *cols.values(), id_)


def init() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn().executescript(SCHEMA)


def source_dir(source_id: int) -> Path:
    return SOURCES_DIR / str(source_id)


# ───────── Tracks as block files ─────────

def add_segment(track_id: int, start: int, wave: np.ndarray) -> None:
    tr = one("SELECT source_id, (SELECT COUNT(*) FROM segments WHERE track_id = tracks.id) AS k FROM tracks WHERE id = ?", track_id)
    if tr is None:
        raise Cancelled()  # track was removed while being produced
    path = write_wav(source_dir(tr["source_id"]) / "tracks" / str(track_id) / f"{tr['k']:05d}.wav", wave)
    insert("segments", track_id=track_id, start=start, n=len(wave), path=str(path), peaks=peaks(wave))
    update("tracks", track_id, covered=start + len(wave))


def read_range(track_id: int, a: int, b: int, cancel: threading.Event | None = None, wait: bool = True) -> np.ndarray:
    """Samples [a, b) of a track. With wait=True, blocks while the track is still being produced."""
    while wait:
        tr = one("SELECT status, covered FROM tracks WHERE id = ?", track_id)
        if tr is None:
            raise RuntimeError("Input track was removed")
        if tr["covered"] >= b or tr["status"] == "complete":
            break
        if tr["status"] != "partial":
            raise RuntimeError(f"Input track {tr['status']}")
        check(cancel)
        time.sleep(0.3)
    out = np.zeros(b - a, np.float32)
    for s in q("SELECT start, n, path FROM segments WHERE track_id = ? AND start < ? AND start + n > ? ORDER BY start", track_id, b, a):
        lo, hi = max(a, s["start"]), min(b, s["start"] + s["n"])
        out[lo - a : hi - a] = read_wav(Path(s["path"]), lo - s["start"], hi - s["start"])
    return out


def iter_blocks(total: int, first: float, size: float):
    """(a, b) sample ranges: one short first block, then fixed-size blocks."""
    a = 0
    while a < total:
        b = min(total, a + int((first if a == 0 else size) * SR))
        yield a, b
        a = b


def delete_track(track_id: int) -> None:
    tr = one("SELECT source_id FROM tracks WHERE id = ?", track_id)
    if tr:
        ex("DELETE FROM tracks WHERE id = ?", track_id)
        shutil.rmtree(source_dir(tr["source_id"]) / "tracks" / str(track_id), ignore_errors=True)


def delete_source(source_id: int) -> None:
    ex("DELETE FROM sources WHERE id = ?", source_id)
    shutil.rmtree(source_dir(source_id), ignore_errors=True)
