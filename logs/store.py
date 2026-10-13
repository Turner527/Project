"""SQLite-журнал запросов пилота: вопрос, вызовы инструментов, ответ, верификация, задержка, токены, оценка."""
import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "logs" / "pilot.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id                  TEXT PRIMARY KEY,
    ts                  TEXT NOT NULL,
    client_id           TEXT,
    question            TEXT NOT NULL,
    tool_calls          TEXT,
    answer              TEXT,
    verification_status TEXT,
    unverified          TEXT,
    latency_ms          INTEGER,
    tokens_prompt       INTEGER,
    tokens_completion   INTEGER,
    tokens_total        INTEGER,
    rating              INTEGER,
    comment             TEXT,
    rated_at            TEXT,
    prompt_version      TEXT,
    llm                 TEXT,
    error               TEXT
)
"""
STATUS_ERROR = "ошибка"


def db_path() -> Path:
    return Path(os.getenv("LOG_DB_PATH") or DEFAULT_DB)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LogStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def log_request(self, result, client_id: str | None = None) -> str:
        """result — AgentResult. Возвращает id записи (по нему потом приходит оценка пользователя)."""
        data = result.to_dict()
        ver = data["verification"]
        request_id = uuid.uuid4().hex
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "INSERT INTO requests (id, ts, client_id, question, tool_calls, answer, verification_status, "
                "unverified, latency_ms, tokens_prompt, tokens_completion, tokens_total, prompt_version, llm, error) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    request_id,
                    _now(),
                    client_id,
                    data["question"],
                    json.dumps(data["tool_calls"], ensure_ascii=False),
                    json.dumps(data["memo"], ensure_ascii=False) if data["memo"] else None,
                    ver["status"] if ver else STATUS_ERROR,
                    json.dumps(ver["unverified"], ensure_ascii=False) if ver else None,
                    data["latency_ms"],
                    data["tokens"]["prompt"],
                    data["tokens"]["completion"],
                    data["tokens"]["total"],
                    data["prompt_version"],
                    data["llm"],
                    data["error"],
                ),
            )
        return request_id

    def set_feedback(self, request_id: str, rating: int | None = None, comment: str | None = None) -> bool:
        """rating: 1 — полезно, -1 — не полезно. Каждое из полей обновляется независимо."""
        if rating not in (None, 1, -1):
            raise ValueError("rating должен быть 1, -1 или None")
        sets, args = [], []
        if rating is not None:
            sets += ["rating = ?", "rated_at = ?"]
            args += [rating, _now()]
        if comment is not None:
            sets.append("comment = ?")
            args.append(comment.strip()[:2000])
        if not sets:
            return False
        with closing(self._conn()) as conn, conn:
            cur = conn.execute(f"UPDATE requests SET {', '.join(sets)} WHERE id = ?", (*args, request_id))
            return cur.rowcount == 1

    def get(self, request_id: str) -> dict | None:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
        return dict(row) if row else None
