from __future__ import annotations

import os
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    stage TEXT NOT NULL DEFAULT 'Waiting',
    prompt TEXT NOT NULL,
    negative_prompt TEXT NOT NULL DEFAULT '',
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    guidance REAL NOT NULL,
    seed INTEGER NOT NULL,
    scheduler TEXT NOT NULL DEFAULT 'linear',
    input_path TEXT,
    image_strength REAL,
    pid_decode INTEGER NOT NULL DEFAULT 0,
    pid_degrade_sigma REAL NOT NULL DEFAULT 0.0,
    output_path TEXT,
    error TEXT,
    progress_current INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    delete_requested INTEGER NOT NULL DEFAULT 0,
    worker_pid INTEGER
);

CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs(status, delete_requested, created_at);

CREATE TABLE IF NOT EXISTS worker_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    pid INTEGER,
    current_job_id TEXT,
    heartbeat REAL,
    model_status TEXT NOT NULL DEFAULT 'Not loaded',
    model_error TEXT
);

INSERT OR IGNORE INTO worker_state(singleton, model_status) VALUES (1, 'Not loaded');
"""


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.database_path

    def initialize(self) -> None:
        self.settings.create_directories()
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA secure_delete = ON")
        try:
            yield conn
        finally:
            conn.close()

    def create_job(self, values: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        job_id = str(uuid.uuid4())
        record = {
            "id": job_id,
            "created_at": now,
            "updated_at": now,
            "status": "queued",
            "stage": "Waiting",
            "progress_total": values["steps"],
            **values,
        }
        columns = ", ".join(record)
        placeholders = ", ".join("?" for _ in record)
        with self.connect() as conn:
            conn.execute(f"INSERT INTO jobs ({columns}) VALUES ({placeholders})", tuple(record.values()))
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def state(self) -> dict[str, Any]:
        with self.connect() as conn:
            queue_rows = conn.execute(
                """SELECT * FROM jobs
                   WHERE status IN ('queued', 'running') AND delete_requested = 0
                   ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at"""
            ).fetchall()
            history_rows = conn.execute(
                """SELECT * FROM jobs
                   WHERE status IN ('completed', 'failed', 'cancelled') AND delete_requested = 0
                   ORDER BY COALESCE(finished_at, updated_at) DESC"""
            ).fetchall()
            worker = conn.execute("SELECT * FROM worker_state WHERE singleton = 1").fetchone()
        return {
            "queue": [dict(row) for row in queue_rows],
            "history": [dict(row) for row in history_rows],
            "worker": dict(worker) if worker else {},
        }

    def claim_next(self, worker_pid: int) -> dict[str, Any] | None:
        now = time.time()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT id FROM jobs
                   WHERE status = 'queued' AND cancel_requested = 0 AND delete_requested = 0
                   ORDER BY created_at LIMIT 1"""
            ).fetchone()
            if not row:
                conn.commit()
                return None
            conn.execute(
                """UPDATE jobs SET status = 'running', stage = 'Loading model', started_at = ?,
                   updated_at = ?, worker_pid = ?, progress_current = 0 WHERE id = ?""",
                (now, now, worker_pid, row["id"]),
            )
            claimed = conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
            conn.commit()
        return dict(claimed)

    def update_progress(self, job_id: str, current: int, total: int, stage: str = "Generating") -> bool:
        now = time.time()
        with self.connect() as conn:
            cursor = conn.execute(
                """UPDATE jobs SET progress_current = ?, progress_total = ?, stage = ?, updated_at = ?
                   WHERE id = ? AND status = 'running'""",
                (current, total, stage, now, job_id),
            )
            row = conn.execute(
                "SELECT cancel_requested, delete_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return cursor.rowcount == 0 or row is None or bool(row["cancel_requested"] or row["delete_requested"])

    def is_cancel_requested(self, job_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested, delete_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return row is None or bool(row["cancel_requested"] or row["delete_requested"])

    def finish(self, job_id: str, status: str, *, output_path: str | None = None, error: str | None = None) -> None:
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """UPDATE jobs SET status = ?, stage = ?, output_path = ?, error = ?, finished_at = ?,
                   updated_at = ?, worker_pid = NULL,
                   progress_current = CASE WHEN ? = 'completed' THEN progress_total ELSE progress_current END
                   WHERE id = ?""",
                (status, status.title(), output_path, error, now, now, status, job_id),
            )

    def request_cancel(self, job_id: str) -> tuple[dict[str, Any] | None, bool]:
        now = time.time()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                conn.commit()
                return None, False
            active = row["status"] == "running"
            if row["status"] == "queued":
                conn.execute(
                    """UPDATE jobs SET status = 'cancelled', stage = 'Cancelled', cancel_requested = 1,
                       finished_at = ?, updated_at = ? WHERE id = ?""",
                    (now, now, job_id),
                )
            elif active:
                conn.execute(
                    "UPDATE jobs SET cancel_requested = 1, stage = 'Cancelling', updated_at = ? WHERE id = ?",
                    (now, job_id),
                )
            updated = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            conn.commit()
        return dict(updated), active

    def request_delete(self, job_id: str) -> tuple[bool, int | None]:
        """Delete immediately unless active; active jobs are hidden and purged by the worker."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                conn.commit()
                return False, None
            if row["status"] == "running":
                conn.execute(
                    """UPDATE jobs SET delete_requested = 1, cancel_requested = 1,
                       stage = 'Deleting', updated_at = ? WHERE id = ?""",
                    (time.time(), job_id),
                )
                conn.commit()
                return True, row["worker_pid"]
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
        self._remove_job_files(dict(row))
        self._truncate_wal()
        return True, None

    def purge(self, job_id: str) -> None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row:
                conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
        if row:
            self._remove_job_files(dict(row))
            self._truncate_wal()

    def recover_incomplete(self) -> None:
        with self.connect() as conn:
            deleting = conn.execute("SELECT id FROM jobs WHERE delete_requested = 1").fetchall()
        for row in deleting:
            self.purge(row["id"])
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """UPDATE jobs SET status = CASE WHEN cancel_requested = 1 THEN 'cancelled' ELSE 'queued' END,
                   stage = CASE WHEN cancel_requested = 1 THEN 'Cancelled' ELSE 'Recovered after restart' END,
                   worker_pid = NULL, updated_at = ?, finished_at = CASE WHEN cancel_requested = 1 THEN ? ELSE NULL END
                   WHERE status = 'running'""",
                (now, now),
            )

    def set_worker_state(
        self,
        *,
        pid: int | None,
        current_job_id: str | None,
        model_status: str,
        model_error: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE worker_state SET pid = ?, current_job_id = ?, heartbeat = ?,
                   model_status = ?, model_error = ? WHERE singleton = 1""",
                (pid, current_job_id, time.time(), model_status, model_error),
            )

    def heartbeat(self, pid: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE worker_state SET heartbeat = ? WHERE singleton = 1 AND pid = ?",
                (time.time(), pid),
            )

    @staticmethod
    def _remove_job_files(job: dict[str, Any]) -> None:
        for key in ("input_path", "output_path"):
            if value := job.get(key):
                try:
                    Path(value).unlink(missing_ok=True)
                except OSError:
                    pass
        output = job.get("output_path")
        if output:
            Path(f"{output}.json").unlink(missing_ok=True)

    def _truncate_wal(self) -> None:
        # secure_delete overwrites the freed row; truncating the WAL also drops old
        # committed frames instead of retaining a soft-deleted copy there.
        with self.connect() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def signal_worker_for_job(db: Database, job_id: str) -> bool:
    with db.connect() as conn:
        state = conn.execute("SELECT pid, current_job_id FROM worker_state WHERE singleton = 1").fetchone()
    if not state or state["current_job_id"] != job_id or not state["pid"]:
        return False
    try:
        os.kill(int(state["pid"]), 2)  # SIGINT interrupts MLX at the next safe point.
    except (OSError, ProcessLookupError):
        return False
    return True
