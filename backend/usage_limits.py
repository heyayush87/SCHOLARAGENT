"""Server-side question limits that protect the demo's API budget.

Two caps, both stored in a small SQLite file so they hold across browser tabs, refreshes and visitors:
  - a shared daily cap for the whole app (the real budget guard: it can't be bypassed by a new browser or URL), and
  - a per-visitor daily cap, so each visitor (e.g. a recruiter trying the demo) gets a few questions.
"""
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

DAILY_QUESTION_LIMIT = int(os.getenv("DAILY_QUESTION_LIMIT", "20"))  # all visitors combined, per UTC day
VISITOR_DAILY_LIMIT = int(os.getenv("VISITOR_DAILY_LIMIT", "4"))  # per visitor, per UTC day
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))

_DB_PATH = Path(os.getenv("USAGE_DB_PATH", Path(__file__).resolve().parent.parent / "usage.db"))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=10, isolation_level=None)  # autocommit; transactions are explicit
    conn.execute("CREATE TABLE IF NOT EXISTS questions (visitor TEXT NOT NULL, ts REAL NOT NULL, day TEXT NOT NULL)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_day ON questions (day)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_visitor_day ON questions (visitor, day)")
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
        used_by_visitor = conn.execute(
            "SELECT COUNT(*) FROM questions WHERE visitor = ? AND day = ?", (visitor, day)
        ).fetchone()[0]
        if used_by_visitor >= VISITOR_DAILY_LIMIT:
            conn.execute("ROLLBACK")
            return (
                f"You've used your {VISITOR_DAILY_LIMIT} demo questions for today. Thanks for trying SCHOLARAGENT! "
                "The README has screenshots and the full code if you'd like to see more."
            )
        conn.execute("INSERT INTO questions (visitor, ts, day) VALUES (?, ?, ?)", (visitor, now, day))
        conn.execute("DELETE FROM questions WHERE day < ?", (day,))  # prune earlier days
        conn.execute("COMMIT")
        return None
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
