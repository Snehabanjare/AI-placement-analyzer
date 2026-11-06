from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path

from history import get_recent_history, init_history_db, record_analysis


class AnalysisHistoryTests(unittest.TestCase):
    def test_records_only_timestamps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "history.db")
            init_history_db(db_path)

            record_analysis(db_path)
            history = get_recent_history(db_path)

            self.assertEqual(len(history), 1)
            with closing(sqlite3.connect(db_path)) as conn:
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(analysis_history)")
                }
            self.assertEqual(columns, {"id", "created_at"})

    def test_migration_removes_legacy_resume_analysis_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "history.db")
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE analysis_history (
                        id INTEGER PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        overall_summary TEXT NOT NULL,
                        matching_skills TEXT NOT NULL,
                        missing_skills TEXT NOT NULL,
                        experience_gaps TEXT NOT NULL,
                        recommendations TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO analysis_history VALUES (
                        1, '2026-01-01T00:00:00+00:00', 'resume.pdf', 'private summary',
                        '[]', '[]', '[]', '[]'
                    )
                    """
                )

            init_history_db(db_path)

            self.assertEqual(get_recent_history(db_path), [])
            with closing(sqlite3.connect(db_path)) as conn:
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(analysis_history)")
                }
            self.assertEqual(columns, {"id", "created_at"})


if __name__ == "__main__":
    unittest.main()
