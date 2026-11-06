from contextlib import closing
import sqlite3
from datetime import datetime, timezone


def init_history_db(db_path: str) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_history'"
        ).fetchone()
        if table:
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(analysis_history)").fetchall()
            }
            if columns != {"id", "created_at"}:
                conn.execute("DROP TABLE analysis_history")
                conn.commit()
                conn.execute("VACUUM")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def record_analysis(db_path: str) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO analysis_history (created_at) VALUES (?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()


def get_recent_history(db_path: str, limit: int = 5) -> list[str]:
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT created_at FROM analysis_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row[0] for row in rows]
