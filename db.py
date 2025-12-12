import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

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
            solver_agent,
            team_agents,
            team_profile,
            critique,
            critique_tags,
            final,
            raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts,
            task,
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