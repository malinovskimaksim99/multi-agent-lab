import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# Шлях до файлу БД
DB_PATH = Path(__file__).parent / "runs.db"


def get_connection() -> sqlite3.Connection:
    """Повертає підключення до SQLite (створює файл, якщо його ще немає)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Створює таблиці, якщо їх ще немає."""
    conn = get_connection()
    cur = conn.cursor()

    # Основна таблиця запусків
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            task TEXT,
            task_type TEXT,
            solver_agent TEXT,
            team_agents TEXT,
            team_profile TEXT,
            critique TEXT,
            critique_tags TEXT,
            final TEXT,
            raw_json TEXT
        )
        """
    )

    # Якщо таблиця вже існувала раніше без task_type – додамо цю колонку
    cur.execute("PRAGMA table_info(runs)")
    columns = [row[1] for row in cur.fetchall()]
    if "task_type" not in columns:
        cur.execute("ALTER TABLE runs ADD COLUMN task_type TEXT")

    # Таблиця прикладів для датасету
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            label TEXT,
            note TEXT,
            created_at TEXT,
            UNIQUE(run_id, label)
        )
        """
    )

    # Таблиця м'яких налаштувань агентів
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            config_key TEXT NOT NULL,
            config_value TEXT NOT NULL,
            updated_at TEXT,
            UNIQUE(agent_name, config_key)
        )
        """
    )

    conn.commit()
    conn.close()


def save_run_to_db(result: Dict[str, Any]) -> None:
    """
    Зберігає результат запуску Supervisor.run(...) в БД.

    Очікується структура, подібна до тієї, що ми записуємо в logs.jsonl:
    {
        "ts": "...",
        "task": "...",
        "solver_agent": "...",
        "team_agents": [...],
        "team_profile": "...",
        "critique": "...",
        "critique_tags": [...],
        "final": "...",
        ...
    }
    """
    # Підстрахуємося: не впасти, якщо чогось немає
    ts = result.get("ts") or ""
    task = result.get("task") or ""
    task_type = result.get("task_type") or ""
    solver_agent = result.get("solver_agent") or ""
    team_agents = result.get("team_agents") or []
    team_profile = result.get("team_profile") or ""
    critique = result.get("critique") or ""
    critique_tags = result.get("critique_tags") or []
    final = result.get("final") or ""

    # Перетворимо складні поля в JSON-рядки
    team_agents_json = json.dumps(team_agents, ensure_ascii=False)
    critique_tags_json = json.dumps(critique_tags, ensure_ascii=False)
    raw_json = json.dumps(result, ensure_ascii=False)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO runs (
            ts,
            task,
            task_type,
            solver_agent,
            team_agents,
            team_profile,
            critique,
            critique_tags,
            final,
            raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts,
            task,
            task_type,
            solver_agent,
            team_agents_json,
            team_profile,
            critique,
            critique_tags_json,
            final,
            raw_json,
        ),
    )

    conn.commit()
    conn.close()


def get_recent_runs(limit: int = 20) -> List[Dict[str, Any]]:
    """Повертає останні N запусків у зручному вигляді."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            id,
            ts,
            task,
            solver_agent,
            team_agents,
            team_profile,
            critique,
            critique_tags,
            final
        FROM runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    conn.close()

    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "id": r["id"],
                "ts": r["ts"],
                "task": r["task"],
                "solver_agent": r["solver_agent"],
                "team_agents": json.loads(r["team_agents"] or "[]"),
                "team_profile": r["team_profile"],
                "critique": r["critique"],
                "critique_tags": json.loads(r["critique_tags"] or "[]"),
                "final": r["final"],
            }
        )
    return results


def mark_run_as_example(run_id: int, label: str = "good", note: Optional[str] = None) -> None:
    """
    Позначає запуск як приклад для датасету.

    label: 'good', 'bad', 'interesting' або будь-який інший текстовий тегі.
    note: довільний коментар (чому цей приклад важливий).
    """
    conn = get_connection()
    cur = conn.cursor()

    created_at = datetime.now(timezone.utc).isoformat()

    cur.execute(
        """
        INSERT OR REPLACE INTO dataset_examples (
            run_id,
            label,
            note,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (run_id, label, note or "", created_at),
    )

    conn.commit()
    conn.close()


def get_dataset_examples(label: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Повертає позначені приклади з таблиці dataset_examples.

    Якщо label вказано – фільтрує за конкретним тегом (наприклад, 'good' чи 'bad').
    """
    conn = get_connection()
    cur = conn.cursor()

    if label:
        cur.execute(
            """
            SELECT
                e.id AS example_id,
                e.run_id,
                e.label,
                e.note,
                e.created_at,
                r.task,
                r.solver_agent,
                r.critique_tags,
                r.final
            FROM dataset_examples e
            JOIN runs r ON r.id = e.run_id
            WHERE e.label = ?
            ORDER BY e.id DESC
            LIMIT ?
            """,
            (label, limit),
        )
    else:
        cur.execute(
            """
            SELECT
                e.id AS example_id,
                e.run_id,
                e.label,
                e.note,
                e.created_at,
                r.task,
                r.solver_agent,
                r.critique_tags,
                r.final
            FROM dataset_examples e
            JOIN runs r ON r.id = e.run_id
            ORDER BY e.id DESC
            LIMIT ?
            """,
            (limit,),
        )

    rows = cur.fetchall()
    conn.close()

    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "example_id": r["example_id"],
                "run_id": r["run_id"],
                "label": r["label"],
                "note": r["note"],
                "created_at": r["created_at"],
                "task": r["task"],
                "solver_agent": r["solver_agent"],
                "critique_tags": json.loads(r["critique_tags"] or "[]"),
                "final": r["final"],
            }
        )
    return results


def set_agent_config(agent_name: str, key: str, value: Any) -> None:
    """
    Зберігає м'які налаштування агента у таблиці agent_configs.

    value серіалізується як JSON-рядок.
    """
    conn = get_connection()
    cur = conn.cursor()

    updated_at = datetime.now(timezone.utc).isoformat()
    value_json = json.dumps(value, ensure_ascii=False)

    cur.execute(
        """
        INSERT OR REPLACE INTO agent_configs (
            agent_name,
            config_key,
            config_value,
            updated_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (agent_name, key, value_json, updated_at),
    )

    conn.commit()
    conn.close()


def get_agent_config(agent_name: str, key: str, default: Any = None) -> Any:
    """
    Повертає значення конкретного налаштування агента або default, якщо його немає.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT config_value
        FROM agent_configs
        WHERE agent_name = ? AND config_key = ?
        """,
        (agent_name, key),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return default

    try:
        return json.loads(row["config_value"])
    except Exception:
        return default


def get_agent_configs(agent_name: str) -> Dict[str, Any]:
    """
    Повертає всі налаштування агента як dict: {key: value}.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT config_key, config_value
        FROM agent_configs
        WHERE agent_name = ?
        """,
        (agent_name,),
    )

    rows = cur.fetchall()
    conn.close()

    configs: Dict[str, Any] = {}
    for r in rows:
        key = r["config_key"]
        try:
            configs[key] = json.loads(r["config_value"])
        except Exception:
            continue
    return configs

init_db()