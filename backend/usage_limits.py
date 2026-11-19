"""Server-side question limits that protect the demo's API budget.

Two caps, both stored in a small SQLite file so they hold across browser tabs, refreshes and visitors:
  - a shared daily cap for the whole app (the real budget guard: it can't be bypassed by a new browser or URL), and
  - a per-visitor rolling hourly cap, so one person can't burn the whole day's allowance.
"""
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

DAILY_QUESTION_LIMIT = int(os.getenv("DAILY_QUESTION_LIMIT", "100"))  # all visitors combined, per UTC day
VISITOR_HOURLY_LIMIT = int(os.getenv("VISITOR_HOURLY_LIMIT", "15"))  # per visitor, rolling 60 minutes
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))

_DB_PATH = Path(os.getenv("USAGE_DB_PATH", Path(__file__).resolve().parent.parent / "usage.db"))
_HOUR = 3600


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=10, isolation_level=None)  # autocommit; transactions are explicit
    conn.execute("CREATE TABLE IF NOT EXISTS questions (visitor TEXT NOT NULL, ts REAL NOT NULL, day TEXT NOT NULL)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_day ON questions (day)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_visitor_ts ON questions (visitor, ts)")
    return conn


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def try_consume(visitor: str) -> str | None:
    """Record one question for `visitor` if both caps allow it. Returns None if allowed, else a reason to show."""
    now, day = time.time(), _today()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")  # lock so concurrent visitors can't both slip past the cap
        used_today = conn.execute("SELECT COUNT(*) FROM questions WHERE day = ?", (day,)).fetchone()[0]
        if used_today >= DAILY_QUESTION_LIMIT:
            conn.execute("ROLLBACK")
            return (
                f"The demo has reached its daily limit of {DAILY_QUESTION_LIMIT} questions across all visitors. "
                "It resets at midnight UTC."
            )
        oldest_in_window = conn.execute(
            "SELECT MIN(ts), COUNT(*) FROM questions WHERE visitor = ? AND ts > ?", (visitor, now - _HOUR)
        ).fetchone()
        if oldest_in_window[1] >= VISITOR_HOURLY_LIMIT:
            conn.execute("ROLLBACK")
            wait_min = max(1, round((oldest_in_window[0] + _HOUR - now) / 60))
            return (
                f"You've asked {VISITOR_HOURLY_LIMIT} questions in the last hour, which is the per-visitor limit. "
                f"Please try again in about {wait_min} minute{'s' if wait_min != 1 else ''}."
            )
        conn.execute("INSERT INTO questions (visitor, ts, day) VALUES (?, ?, ?)", (visitor, now, day))
        conn.execute("DELETE FROM questions WHERE day < ? AND ts < ?", (day, now - _HOUR))  # prune old rows
        conn.execute("COMMIT")
        return None
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
