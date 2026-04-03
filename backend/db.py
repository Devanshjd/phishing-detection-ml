# backend/db.py
"""SQLite scan history storage."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path(__file__).resolve().parent / "history.db"


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_type   TEXT    NOT NULL,
                input_value TEXT    NOT NULL,
                verdict     TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                meta_json   TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            )
            """
        )
        con.commit()


def insert_scan(
    scan_type: str,
    input_value: str,
    verdict: str,
    confidence: float,
    meta_json: str,
    created_at: str,
) -> int:
    with _connect() as con:
        cur = con.execute(
            """
            INSERT INTO scans (scan_type, input_value, verdict, confidence, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (scan_type, input_value[:10_000], verdict, confidence, meta_json, created_at),
        )
        con.commit()
        return int(cur.lastrowid)


def get_recent(limit: int = 20) -> List[Dict[str, Any]]:
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_stats() -> Dict[str, Any]:
    """Aggregate statistics for the dashboard."""
    with _connect() as con:
        total = con.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        phishing = con.execute(
            "SELECT COUNT(*) FROM scans WHERE verdict IN ('phishing', 'suspicious')"
        ).fetchone()[0]
        by_type = {}
        for row in con.execute(
            "SELECT scan_type, COUNT(*) as cnt FROM scans GROUP BY scan_type"
        ):
            by_type[row[0]] = row[1]
        return {
            "total_scans": total,
            "threats_detected": phishing,
            "by_type": by_type,
        }
