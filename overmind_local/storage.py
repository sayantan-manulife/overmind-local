"""SQLite-backed storage — no cloud, no API key required."""
import sqlite3
import json
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

_db_path: Optional[Path] = None


def init(base_dir: str = ".overmind") -> Path:
    global _db_path
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    _db_path = base / "traces.db"
    _create_schema(_db_path)
    return _db_path


def get_db_path() -> Path:
    if _db_path is None:
        raise RuntimeError("Call overmind_local.init() before using storage")
    return _db_path


def _create_schema(path: Path):
    with sqlite3.connect(str(path)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS spans (
                id          TEXT PRIMARY KEY,
                trace_id    TEXT NOT NULL,
                parent_id   TEXT,
                name        TEXT NOT NULL,
                span_type   TEXT,
                agent_name  TEXT,
                input       TEXT,
                output      TEXT,
                error       TEXT,
                start_time  REAL,
                end_time    REAL,
                duration_ms REAL,
                metadata    TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_spans_trace   ON spans(trace_id);
            CREATE INDEX IF NOT EXISTS idx_spans_agent   ON spans(agent_name);
            CREATE INDEX IF NOT EXISTS idx_spans_created ON spans(created_at);

            CREATE TABLE IF NOT EXISTS policies (
                id          TEXT PRIMARY KEY,
                agent_name  TEXT NOT NULL,
                name        TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS dataset_items (
                id              TEXT PRIMARY KEY,
                agent_name      TEXT NOT NULL,
                input           TEXT,
                expected_output TEXT,
                metadata        TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );
        """)


def write_span(span: Dict[str, Any], db_path: Optional[Path] = None):
    path = db_path or get_db_path()
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO spans
               (id, trace_id, parent_id, name, span_type, agent_name,
                input, output, error, start_time, end_time, duration_ms, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                span["id"], span["trace_id"], span.get("parent_id"),
                span["name"], span.get("span_type"), span.get("agent_name"),
                _j(span.get("input")), _j(span.get("output")), span.get("error"),
                span.get("start_time"), span.get("end_time"), span.get("duration_ms"),
                _j(span.get("metadata", {})),
            ),
        )


def get_spans(agent_name: Optional[str] = None, limit: int = 100,
              db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = db_path or get_db_path()
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        q = "SELECT * FROM spans WHERE 1=1"
        p: list = []
        if agent_name:
            q += " AND agent_name = ?"
            p.append(agent_name)
        q += " ORDER BY start_time DESC LIMIT ?"
        p.append(limit)
        return [_deserialize_span(dict(r)) for r in conn.execute(q, p).fetchall()]


def _deserialize_span(row: Dict[str, Any]) -> Dict[str, Any]:
    for field in ("input", "output", "metadata"):
        val = row.get(field)
        if val:
            try:
                row[field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
    return row


def get_policies(agent_name: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = db_path or get_db_path()
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM policies WHERE agent_name = ? ORDER BY created_at",
            [agent_name],
        ).fetchall()]


def add_policy(agent_name: str, name: str, description: str,
               db_path: Optional[Path] = None):
    path = db_path or get_db_path()
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "INSERT INTO policies (id, agent_name, name, description) VALUES (?,?,?,?)",
            [str(uuid.uuid4()), agent_name, name, description],
        )


def add_dataset_item(agent_name: str, input_data: dict,
                     expected_output: Optional[str] = None,
                     metadata: Optional[dict] = None,
                     db_path: Optional[Path] = None):
    path = db_path or get_db_path()
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """INSERT INTO dataset_items (id, agent_name, input, expected_output, metadata)
               VALUES (?,?,?,?,?)""",
            [str(uuid.uuid4()), agent_name, _j(input_data),
             expected_output, _j(metadata or {})],
        )


def get_dataset(agent_name: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = db_path or get_db_path()
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM dataset_items WHERE agent_name = ? ORDER BY created_at",
            [agent_name],
        ).fetchall()]


def _j(obj) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(obj)
    except Exception:
        return json.dumps(str(obj))
