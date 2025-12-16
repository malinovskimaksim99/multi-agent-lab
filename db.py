import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
            project TEXT,
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

    # Якщо таблиця вже існувала раніше без task_type / project – додамо ці колонки
    cur.execute("PRAGMA table_info(runs)")
    columns = [row[1] for row in cur.fetchall()]
    if "task_type" not in columns:
        cur.execute("ALTER TABLE runs ADD COLUMN task_type TEXT")
    if "project" not in columns:
        cur.execute("ALTER TABLE runs ADD COLUMN project TEXT DEFAULT 'default'")

    # Таблиця прикладів для датасету
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            project TEXT,
            label TEXT,
            note TEXT,
            created_at TEXT,
            UNIQUE(run_id, label)
        )
        """
    )

    # Міграція для dataset_examples.project
    cur.execute("PRAGMA table_info(dataset_examples)")
    ds_columns = [row[1] for row in cur.fetchall()]
    if "project" not in ds_columns:
        cur.execute("ALTER TABLE dataset_examples ADD COLUMN project TEXT DEFAULT 'default'")

    # Таблиця м'яких налаштувань агентів
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            project TEXT,
            config_key TEXT NOT NULL,
            config_value TEXT NOT NULL,
            updated_at TEXT,
            UNIQUE(agent_name, config_key, project)
        )
        """
    )

    # Міграція для agent_configs.project
    cur.execute("PRAGMA table_info(agent_configs)")
    cfg_columns = [row[1] for row in cur.fetchall()]
    if "project" not in cfg_columns:
        cur.execute("ALTER TABLE agent_configs ADD COLUMN project TEXT DEFAULT 'default'")

    # Таблиця проєктів
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            type TEXT,
            description TEXT,
            status TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )

    # Таблиця налаштувань проєктів (key/value, як для конфігів агентів)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS project_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT,
            UNIQUE(project_id, key)
        )
        """
    )

    # Таблиця загального стану застосунку (поточний проєкт тощо)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def _auto_label_dataset_example(
    task_type: str,
    critique_tags: List[Any],
    final_text: str,
) -> Optional[Tuple[str, str]]:
    """
    Повертає (label, note) для датасету або None, якщо приклад не варто додавати.

    Логіка (перша повноцінна версія):
      - якщо є явно негативні теги критика -> 'bad';
      - якщо відповідь дуже коротка або взагалі порожня -> 'bad';
      - якщо немає негативних тегів, відповідь достатньо розгорнута
        і задача відноситься до корисних типів -> 'good';
      - інакше не додаємо в датасет (None).
    """
    negative = {
        "too_short",
        "missing_structure",
        "missing_steps",
        "incorrect",
        "off_topic",
    }

    # Нормалізуємо теги до простого списку рядків
    tags = [str(t) for t in (critique_tags or [])]

    # 1) Явно погані кейси
    if any(t in negative for t in tags):
        return "bad", f"auto: negative critique tags -> {tags}"

    # 2) Дуже коротка / порожня відповідь
    text = (final_text or "").strip()
    if not text:
        return "bad", "auto: empty final answer"
    word_count = len(text.split())
    if word_count < 20:
        return "bad", f"auto: very short answer ({word_count} words)"

    # 3) Потенційно хороші приклади
    useful_types = {"code", "plan", "docs", "db_analysis", "meta", "explain"}
    normalized_type = task_type or "other"
    if normalized_type in useful_types and not tags:
        return "good", f"auto: clean answer for task_type={normalized_type}"

    # 4) Інакше не додаємо до датасету
    return None


def save_run_to_db(result: Dict[str, Any]) -> None:
    """
    Зберігає результат запуску Supervisor.run(...) в БД і,
    за потреби, додає приклад у dataset_examples.

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
    project = result.get("project") or "default"
    task = result.get("task") or ""
    task_type = result.get("task_type") or "other"
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
            project,
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
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts,
            project,
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

    run_id = cur.lastrowid

    # --- Автоматичне додавання в датасет (якщо варто) ---
    auto = _auto_label_dataset_example(task_type, critique_tags, final)
    if auto is not None:
        label, note = auto
        created_at = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """
            INSERT INTO dataset_examples (run_id, project, label, note, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, project, label, note, created_at),
        )

    conn.commit()
    conn.close()


def get_recent_runs(limit: int = 20, project: Optional[str] = None) -> List[Dict[str, Any]]:
    """Повертає останні N запусків у зручному вигляді (за потреби — по конкретному проєкту)."""
    conn = get_connection()
    cur = conn.cursor()

    if project is not None:
        cur.execute(
            """
            SELECT
                id,
                ts,
                project,
                task,
                task_type,
                solver_agent,
                team_agents,
                team_profile,
                critique,
                critique_tags,
                final
            FROM runs
            WHERE project = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (project, limit),
        )
    else:
        cur.execute(
            """
            SELECT
                id,
                ts,
                project,
                task,
                task_type,
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
                "project": r["project"],
                "task": r["task"],
                "task_type": r["task_type"],
                "solver_agent": r["solver_agent"],
                "team_agents": json.loads(r["team_agents"] or "[]"),
                "team_profile": r["team_profile"],
                "critique": r["critique"],
                "critique_tags": json.loads(r["critique_tags"] or "[]"),
                "final": r["final"],
            }
        )
    return results


def mark_run_as_example(
    run_id: int,
    label: str = "good",
    note: Optional[str] = None,
    project: Optional[str] = None,
) -> None:
    """
    Позначає запуск як приклад для датасету.

    label: 'good', 'bad', 'interesting' або будь-який інший текстовий тегі.
    note: довільний коментар (чому цей приклад важливий).
    """
    conn = get_connection()
    cur = conn.cursor()

    if project is None:
        # спробуємо підтягнути project з таблиці runs
        cur.execute(
            "SELECT project FROM runs WHERE id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        project = row["project"] if row and row["project"] is not None else "default"

    created_at = datetime.now(timezone.utc).isoformat()

    cur.execute(
        """
        INSERT OR REPLACE INTO dataset_examples (
            run_id,
            project,
            label,
            note,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, project, label, note or "", created_at),
    )

    conn.commit()
    conn.close()


def get_dataset_examples(
    label: Optional[str] = None,
    limit: int = 50,
    project: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Повертає позначені приклади з таблиці dataset_examples.

    Якщо label вказано – фільтрує за конкретним тегом (наприклад, 'good' чи 'bad').
    Якщо project вказано – фільтрує за конкретним проєктом.
    """
    conn = get_connection()
    cur = conn.cursor()

    base_select = """
        SELECT
            e.id AS example_id,
            e.run_id,
            e.project,
            e.label,
            e.note,
            e.created_at,
            r.task,
            r.solver_agent,
            r.critique_tags,
            r.final
        FROM dataset_examples e
        JOIN runs r ON r.id = e.run_id
    """
    params: List[Any] = []
    where_clauses: List[str] = []

    if label:
        where_clauses.append("e.label = ?")
        params.append(label)
    if project is not None:
        where_clauses.append("e.project = ?")
        params.append(project)

    if where_clauses:
        base_select += " WHERE " + " AND ".join(where_clauses)

    base_select += " ORDER BY e.id DESC LIMIT ?"
    params.append(limit)

    cur.execute(base_select, tuple(params))

    rows = cur.fetchall()
    conn.close()

    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "example_id": r["example_id"],
                "run_id": r["run_id"],
                "project": r["project"],
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


def set_agent_config(agent_name: str, key: str, value: Any, project: str = "default") -> None:
    """
    Зберігає м'які налаштування агента у таблиці agent_configs.

    value серіалізується як JSON-рядок.
    project - назва проєкту для ізоляції конфігурацій.
    """
    conn = get_connection()
    cur = conn.cursor()

    updated_at = datetime.now(timezone.utc).isoformat()
    value_json = json.dumps(value, ensure_ascii=False)

    cur.execute(
        """
        INSERT OR REPLACE INTO agent_configs (
            agent_name,
            project,
            config_key,
            config_value,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (agent_name, project, key, value_json, updated_at),
    )

    conn.commit()
    conn.close()


def get_agent_config(
    agent_name: str,
    key: str,
    default: Any = None,
    project: str = "default",
) -> Any:
    """
    Повертає значення конкретного налаштування агента або default, якщо його немає.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT config_value
        FROM agent_configs
        WHERE agent_name = ? AND config_key = ? AND project = ?
        """,
        (agent_name, key, project),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return default

    try:
        return json.loads(row["config_value"])
    except Exception:
        return default


def get_agent_configs(agent_name: str, project: str = "default") -> Dict[str, Any]:
    """
    Повертає всі налаштування агента як dict: {key: value}.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT config_key, config_value
        FROM agent_configs
        WHERE agent_name = ? AND project = ?
        """,
        (agent_name, project),
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


# ----------------- Project helpers -----------------

def ensure_default_project() -> str:
    """
    Гарантує, що в таблиці projects є хоча б один активний проєкт
    і що app_state.current_project вказує на існуючий проєкт.

    Повертає назву поточного проєкту.
    """
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    # 1) Пробуємо прочитати поточний проєкт
    cur.execute(
        "SELECT value FROM app_state WHERE key = 'current_project'"
    )
    row = cur.fetchone()
    if row and row["value"]:
        current = str(row["value"])
        # Перевіряємо, чи такий проєкт існує
        cur.execute(
            "SELECT id FROM projects WHERE name = ?",
            (current,),
        )
        prow = cur.fetchone()
        if prow:
            conn.close()
            return current
        # Якщо current_project вказує в нікуди – нижче виберемо коректний проєкт

    # 2) Якщо current_project немає або він битий – шукаємо перший активний проєкт
    cur.execute(
        """
        SELECT name
        FROM projects
        WHERE status = 'active'
        ORDER BY id ASC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if row and row["name"]:
        name = str(row["name"])
    else:
        # 3) Якщо взагалі немає жодного проєкту – створюємо базовий "Розробка"
        name = "Розробка"
        cur.execute(
            """
            INSERT INTO projects (name, type, description, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                "dev",
                "Головний проєкт розробки та навчання multi-agent-lab",
                "active",
                now,
                now,
            ),
        )

    # 4) Оновлюємо current_project у app_state
    cur.execute(
        """
        INSERT OR REPLACE INTO app_state (key, value, updated_at)
        VALUES (?, ?, ?)
        """,
        ("current_project", name, now),
    )

    conn.commit()
    conn.close()
    return name


def create_project(
    name: str,
    type_: str = "generic",
    description: str = "",
    status: str = "active",
) -> None:
    """
    Створює новий проєкт, якщо з таким ім'ям ще немає.
    """
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    cur.execute(
        "SELECT id FROM projects WHERE name = ?",
        (name,),
    )
    row = cur.fetchone()
    if not row:
        cur.execute(
            """
            INSERT INTO projects (name, type, description, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, type_, description, status, now, now),
        )

    conn.commit()
    conn.close()


def get_projects() -> List[Dict[str, Any]]:
    """
    Повертає список усіх проєктів.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, name, type, description, status, created_at, updated_at
        FROM projects
        ORDER BY id ASC
        """
    )
    rows = cur.fetchall()
    conn.close()

    projects: List[Dict[str, Any]] = []
    for r in rows:
        projects.append(
            {
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "description": r["description"],
                "status": r["status"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
        )
    return projects


def get_current_project() -> str:
    """
    Повертає назву поточного проєкту. Якщо нічого не налаштовано – гарантує, що обрано існуючий проєкт (через ensure_default_project).
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT value FROM app_state WHERE key = 'current_project'"
    )
    row = cur.fetchone()
    conn.close()

    if row and row["value"]:
        return str(row["value"])
    # Якщо щось пішло не так – підстрахуємось
    return ensure_default_project()


def set_current_project(name: str) -> None:
    """
    Встановлює поточний проєкт (створює його, якщо ще не існує).
    """
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    # Переконаємось, що проєкт існує
    cur.execute("SELECT id FROM projects WHERE name = ?", (name,))
    row = cur.fetchone()
    if not row:
        cur.execute(
            """
            INSERT INTO projects (name, type, description, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, "generic", "", "active", now, now),
        )

    # Оновлюємо current_project
    cur.execute(
        """
        INSERT OR REPLACE INTO app_state (key, value, updated_at)
        VALUES (?, ?, ?)
        """,
        ("current_project", name, now),
    )

    conn.commit()
    conn.close()

def get_solver_stats_by_task_type(task_type: str) -> Dict[str, int]:
    """
    Повертає статистику по агентам для вказаного типу задач (task_type).

    Якщо task_type == "other" або порожній, повертаємо агреговану статистику
    по всіх запусках без фільтра по типам задач.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        if task_type and task_type != "other":
            cur.execute(
                """
                SELECT solver_agent, COUNT(*) AS cnt
                FROM runs
                WHERE task_type = ?
                GROUP BY solver_agent
                """,
                (task_type,),
            )
        else:
            cur.execute(
                """
                SELECT solver_agent, COUNT(*) AS cnt
                FROM runs
                GROUP BY solver_agent
                """
            )
        rows = cur.fetchall()
        stats: Dict[str, int] = {}
        for solver, cnt in rows:
            key = solver or "unknown"
            stats[key] = int(cnt or 0)
        return stats
    finally:
        conn.close()
init_db()
ensure_default_project()