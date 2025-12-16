import json
import sys
import subprocess
import sqlite3
import re
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime, timezone, timedelta
from db import get_recent_runs, mark_run_as_example, get_dataset_examples, set_agent_config


LOGS = Path("logs.jsonl")


def _safe_read_logs(limit: int = 200) -> List[Dict[str, Any]]:
    if not LOGS.exists():
        return []
    lines = LOGS.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in lines[-limit:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def _parse_ts(obj: Dict[str, Any]) -> Optional[datetime]:
    ts = obj.get("ts")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _filter_by_day(data: List[Dict[str, Any]], day_utc: datetime) -> List[Dict[str, Any]]:
    target_date = day_utc.date()
    filtered = []
    for obj in data:
        dt = _parse_ts(obj)
        if not dt:
            continue
        if dt.date() == target_date:
            filtered.append(obj)
    return filtered


def _resolve_day_from_text(text: str) -> Optional[datetime]:
    t = text.lower()
    now = datetime.now(timezone.utc)

    if "сьогодні" in t:
        return now
    if "вчора" in t:
        return now - timedelta(days=1)

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d, tzinfo=timezone.utc)
        except Exception:
            return None

    return None


def _extract_limit(text: str, default: int = 10, max_limit: int = 100) -> int:
    m = re.search(r"\b(\d{1,3})\b", text)
    if not m:
        return default
    n = int(m.group(1))
    if n <= 0:
        return default
    return min(n, max_limit)


def show_memory() -> str:
    from memory.store import load_memory
    return json.dumps(load_memory(), ensure_ascii=False, indent=2)


def show_agents() -> str:
    from agents.registry import list_agents
    agents_list = list_agents()
    return "Registered agents: " + ", ".join(agents_list)


def show_recent_runs(limit: int = 10, day_filter: Optional[datetime] = None) -> str:
    data = _safe_read_logs(limit=500)
    if not data:
        return "Немає логів запусків поки що."

    if day_filter:
        data = _filter_by_day(data, day_filter)

    if not data:
        return "Немає запусків для заданого періоду."

    tail = data[-limit:]
    lines = []
    for obj in tail:
        task = obj.get("task", "")
        solver = obj.get("solver_agent")
        team = obj.get("team_agents")
        tags = obj.get("critique_tags") or []
        if team:
            lines.append(f"- task: {task} | team: {team} | solver: {solver} | tags: {tags}")
        else:
            lines.append(f"- task: {task} | solver: {solver} | tags: {tags}")
    return "\n".join(lines)


def show_errors(limit: int = 20, day_filter: Optional[datetime] = None) -> str:
    data = _safe_read_logs(limit=800)
    if not data:
        return "Немає логів запусків поки що."

    if day_filter:
        data = _filter_by_day(data, day_filter)

    tagged = []
    for obj in data:
        tags = obj.get("critique_tags") or []
        if tags:
            tagged.append(obj)

    if not tagged:
        return "За останні запуски не знайдено tagged-проблем."

    tail = tagged[-limit:]
    lines = []
    for obj in tail:
        lines.append(f"- {obj.get('task','')} | tags: {obj.get('critique_tags')}")
    return "\n".join(lines)


def show_db_runs(limit: int = 10) -> str:
    """Показати останні запуски з SQLite-БД (runs.db)."""
    try:
        runs = get_recent_runs(limit=limit)
    except Exception as e:
        return f"Не вдалося прочитати запуски з БД: {e}"

    if not runs:
        return "У БД ще немає збережених запусків."

    lines = []
    for r in runs:
        task = r.get("task", "") or ""
        if len(task) > 80:
            task = task[:77] + "..."
        solver = r.get("solver_agent")
        team = r.get("team_agents") or []
        tags = r.get("critique_tags") or []
        ts = r.get("ts", "") or ""
        rid = r.get("id")

        if team:
            lines.append(
                f"- [id={rid}] {ts} | team: {team} | solver: {solver} | tags: {tags} | task: {task}"
            )
        else:
            lines.append(
                f"- [id={rid}] {ts} | solver: {solver} | tags: {tags} | task: {task}"
            )

    return "\n".join(lines)


# --- Projects helpers (SQLite, projects table) ---

def _get_db_connection():
    """Отримати з'єднання з локальною БД runs.db."""
    return sqlite3.connect("runs.db")

def show_projects() -> str:
    """
    Показати список проєктів з таблиці projects.
    Формат: id, name, type, status, created_at.
    """
    try:
        conn = _get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, type, status, created_at, updated_at
            FROM projects
            ORDER BY id ASC
            """
        )
        rows = cur.fetchall()
    except Exception as e:
        return f"Не вдалося прочитати список проєктів з БД: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not rows:
        return "У БД поки немає жодного проєкту."

    lines: List[str] = ["Список проєктів:"]
    for rid, name, ptype, status, created_at, updated_at in rows:
        lines.append(
            f"- id={rid} | name={name} | type={ptype} | status={status} | created_at={created_at}"
        )
    return "\n".join(lines)

def show_current_project() -> str:
    """
    Показати поточний/останній активний проєкт.

    Наразі ми вважаємо поточним той проєкт, у якого найбільший updated_at.
    Якщо updated_at однаковий або порожній, беремо з мінімальним id.
    """
    try:
        conn = _get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, type, status, created_at, updated_at
            FROM projects
            ORDER BY updated_at DESC, id ASC
            LIMIT 1
            """
        )
        row = cur.fetchone()
    except Exception as e:
        return f"Не вдалося визначити поточний проєкт з БД: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not row:
        return "Поточний проєкт не знайдено (таблиця projects порожня)."

    rid, name, ptype, status, created_at, updated_at = row
    return (
        "Поточний проєкт:\n"
        f"- id={rid}\n"
        f"- name={name}\n"
        f"- type={ptype}\n"
        f"- status={status}\n"
        f"- created_at={created_at}\n"
        f"- updated_at={updated_at}"
    )

def _projects_from_text(text: str) -> str:
    """
    Обробка тексту для команд типу:
      - 'проєкти'
      - 'список проєктів'
      - 'projects list'
    """
    return show_projects()

def _current_project_from_text(text: str) -> str:
    """
    Обробка тексту для команд типу:
      - 'поточний проєкт'
      - 'current project'
    """
    return show_current_project()


def _db_runs_from_text(text: str) -> str:
    limit = _extract_limit(text, default=10)
    return show_db_runs(limit=limit)


def _dataset_add_from_text(text: str) -> str:
    """
    Позначити запуск як приклад для датасету.

    Очікується, що в тексті буде хоча б один номер (run_id).
    Можна вказати label через "як <label>", наприклад:
      - "додай запуск 5 в датасет як good"
      - "додай запуск 3 в датасет як bad"
    Якщо label не знайдено, за замовчуванням використовується "good".
    """
    m_id = re.search(r"\b(\d+)\b", text)
    if not m_id:
        return "Не знайшов номер запуску в тексті. Приклад: 'додай запуск 5 в датасет як good'."

    run_id = int(m_id.group(1))

    # Спробуємо витягнути label після "як" або "as"
    m_label = re.search(r"як\s+(\w+)", text.lower())
    if not m_label:
        m_label = re.search(r"as\s+(\w+)", text.lower())

    label = m_label.group(1) if m_label else "good"

    try:
        # Як note збережемо повний текст команди, щоб не загубити контекст
        mark_run_as_example(run_id=run_id, label=label, note=text)
        return f"Запуск {run_id} позначено як приклад у датасеті з міткою '{label}'."
    except Exception as e:
        return f"Не вдалося позначити запуск {run_id} у датасеті: {e}"


def _dataset_show_from_text(text: str) -> str:
    """
    Показати приклади з датасету.

    Можна вказати label у тексті:
      - 'покажи датасет good'
      - 'покажи датасет bad'
      - 'покажи датасет interesting'
    Якщо label не знайдено, показуються приклади з усіма мітками.
    Число в тексті використовується як limit (за замовчуванням 10).
    """
    limit = _extract_limit(text, default=10, max_limit=100)

    t = text.lower()

    # Визначаємо label за ключовими словами, якщо є
    label: Optional[str] = None
    if "good" in t or "гарн" in t:
        label = "good"
    elif "bad" in t or "поган" in t:
        label = "bad"
    elif "interesting" in t or "цікав" in t:
        label = "interesting"

    try:
        examples = get_dataset_examples(label=label, limit=limit)
    except Exception as e:
        return f"Не вдалося прочитати датасет-приклади з БД: {e}"

    if not examples:
        return "У датасеті поки немає позначених прикладів."

    lines: List[str] = []
    for ex in examples:
        task = ex.get("task", "") or ""
        if len(task) > 80:
            task = task[:77] + "..."
        solver = ex.get("solver_agent") or ""
        tags = ex.get("critique_tags") or []
        example_id = ex.get("example_id")
        run_id = ex.get("run_id")
        label_val = ex.get("label") or ""
        note = ex.get("note") or ""
        created_at = ex.get("created_at") or ""

        if note:
            lines.append(
                f"- [example_id={example_id}, run_id={run_id}] {created_at} | label: {label_val} | solver: {solver} | tags: {tags} | task: {task} | note: {note}"
            )
        else:
            lines.append(
                f"- [example_id={example_id}, run_id={run_id}] {created_at} | label: {label_val} | solver: {solver} | tags: {tags} | task: {task}"
            )

    return "\n".join(lines)


def run_progress_report() -> str:
    p = Path("progress_report.py")
    if not p.exists():
        return "progress_report.py не знайдено в проєкті."
    try:
        res = subprocess.run(
            [sys.executable, str(p)],
            capture_output=True,
            text=True,
            check=False
        )
        out = (res.stdout or "").strip()
        err = (res.stderr or "").strip()
        return out if out else (err if err else "Звіт виконано без виводу.")
    except Exception as e:
        return f"Не вдалося запустити progress_report.py: {e}"


def run_eval_runner() -> str:
    p = Path("eval_runner.py")
    if not p.exists():
        return "eval_runner.py не знайдено в проєкті."
    try:
        res = subprocess.run(
            [sys.executable, str(p)],
            capture_output=True,
            text=True,
            check=False
        )
        out = (res.stdout or "").strip()
        err = (res.stderr or "").strip()
        return out if out else (err if err else "Оцінку виконано без виводу.")
    except Exception as e:
        return f"Не вдалося запустити eval_runner.py: {e}"


def run_trainer_analysis(limit: int = 50) -> str:
    """
    Запустити TrainerAgent через app.py, щоб отримати узагальнений аналіз запусків з БД.
    Використовується як бекенд для чат-команди 'аналіз агентів' / 'аналіз запусків'.
    """
    p = Path("app.py")
    if not p.exists():
        return "app.py не знайдено в проєкті."

    task = f"Зроби аналіз {limit} останніх запусків у БД (виклик тренера через команду)."
    try:
        res = subprocess.run(
            [sys.executable, str(p), "--task", task, "--auto"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (res.stdout or "").strip()
        err = (res.stderr or "").strip()
        return out if out else (err if err else "Аналіз виконано без виводу.")
    except Exception as e:
        return f"Не вдалося запустити аналіз тренера: {e}"


def _trainer_from_text(text: str) -> str:
    """
    Обробка тексту для команди аналізу запусків/агентів.
    Число в тексті інтерпретується як 'скільки останніх запусків' аналізувати.
    """
    limit = _extract_limit(text, default=50, max_limit=200)
    return run_trainer_analysis(limit=limit)


# --- NEW: Meta-train full cycle ---
def run_meta_train(limit: int = 50) -> str:
    """
    Запустити повний meta-тренувальний цикл:
      1) аналіз запусків тренером (run_trainer_analysis),
      2) застосування пропозицій тренера до agent_configs (apply_trainer_suggestions).

    Повертає короткий текстовий звіт по обом крокам.
    """
    analysis = run_trainer_analysis(limit=limit)
    apply_result = apply_trainer_suggestions(limit=limit)

    parts: List[str] = []
    parts.append("=== Аналіз тренера ===")
    parts.append(analysis)
    parts.append("")
    parts.append("=== Оновлення конфігів ===")
    parts.append(apply_result)

    return "\n".join(parts)


def _meta_train_from_text(text: str) -> str:
    """
    Команда для запуску meta-тренування:
    - аналіз останніх N запусків;
    - оновлення конфігів агентів за пропозиціями тренера.
    Число в тексті інтерпретується як 'скільки останніх запусків' аналізувати.
    """
    limit = _extract_limit(text, default=50, max_limit=200)
    return run_meta_train(limit=limit)


# --- NEW: Trainer suggestions extraction and apply ---
def _extract_trainer_suggestions(text: str) -> Dict[str, Any]:
    """
    Виділити JSON-блок з пропозиціями конфігів з відповіді TrainerAgent.

    Шукає фрагмент між ```json ... ``` і пробує розпарсити його як dict.
    Якщо не знаходить або парсинг не вдається, повертає {}.
    """
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        return {}
    raw = m.group(1)
    try:
        return json.loads(raw)
    except Exception:
        return {}


def apply_trainer_suggestions(limit: int = 50) -> str:
    """
    Запустити TrainerAgent, витягнути з його відповіді config_suggestions
    та оновити agent_configs у БД через set_agent_config.
    """
    p = Path("app.py")
    if not p.exists():
        return "app.py не знайдено в проєкті."

    task = f"Зроби аналіз {limit} останніх запусків у БД (оновлення конфігів агентів за тренером)."
    try:
        res = subprocess.run(
            [sys.executable, str(p), "--task", task, "--auto"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return f"Не вдалося запустити аналіз тренера: {e}"

    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    text = out if out else err
    if not text:
        return "Не вдалося отримати відповідь тренера."

    suggestions = _extract_trainer_suggestions(text)
    if not suggestions:
        return "Тренер не повернув пропозицій для agent_configs."

    applied: List[str] = []
    errors: List[str] = []

    for agent_name, cfg in suggestions.items():
        if not isinstance(cfg, dict):
            continue
        try:
            # cfg очікується як dict з ключами конфігу, наприклад:
            # {"preferred_task_types": ["db_analysis"]}
            for key, value in cfg.items():
                set_agent_config(agent_name, key, value)
            applied.append(agent_name)
        except Exception as e:
            errors.append(f"{agent_name}: {e}")

    if not applied and not errors:
        return "Не вдалося застосувати жодну пропозицію тренера."

    lines: List[str] = []
    if applied:
        agents_str = ", ".join(sorted(set(applied)))
        lines.append(f"Застосовано пропозиції тренера для агентів: {agents_str}.")
    if errors:
        lines.append("Помилки під час оновлення деяких агентів:\n- " + "\n- ".join(errors))

    return "\n".join(lines)


def _trainer_apply_from_text(text: str) -> str:
    """
    Команда для застосування пропозицій тренера до agent_configs.
    Число в тексті інтерпретується як 'скільки останніх запусків' аналізувати.
    """
    limit = _extract_limit(text, default=50, max_limit=200)
    return apply_trainer_suggestions(limit=limit)


def help_text() -> str:
    return (
        "Доступні живі команди:\n"
        "- зроби звіт / progress report\n"
        "- запусти оцінку / eval\n"
        "- покажи пам’ять / memory\n"
        "- покажи агентів / список агентів\n"
        "- покажи помилки / errors\n"
        "- покажи останні запуски / last runs\n"
        "- покажи запуски з бд / db runs\n"
        "- додай запуск в датасет / add to dataset\n"
        "- покажи датасет / dataset\n"
        "- аналіз агентів / аналіз запусків\n"
        "- список проєктів / projects list\n"
        "- поточний проєкт / current project\n"
        "- meta тренування / meta training\n"
        "- застосуй пропозиції тренера / trainer apply configs\n"
        "\nПараметризовані приклади:\n"
        "- покажи останні 20 запусків\n"
        "- покажи помилки за сьогодні\n"
        "- покажи останні 5 запусків за вчора\n"
        "- аналіз 50 останніх запусків\n"
        "\nSlash-команди:\n"
        "- /memory, /agents, /route <task>"
    )


def _runs_from_text(text: str) -> str:
    limit = _extract_limit(text, default=10)
    day = _resolve_day_from_text(text)
    return show_recent_runs(limit=limit, day_filter=day)


def _errors_from_text(text: str) -> str:
    limit = _extract_limit(text, default=20)
    day = _resolve_day_from_text(text)
    return show_errors(limit=limit, day_filter=day)


def _report_from_text(text: str) -> str:
    return run_progress_report()


def _eval_from_text(text: str) -> str:
    return run_eval_runner()


COMMANDS: List[Dict[str, Any]] = [
    {
        "name": "help",
        "patterns": ["help", "довідка", "команди", "що вмієш", "що можеш"],
        "handler": lambda text: help_text(),
    },
    {
        "name": "report",
        "patterns": ["зроби звіт", "покажи звіт", "progress report", "прогрес"],
        "handler": _report_from_text,
    },
    {
        "name": "eval",
        "patterns": [
            "запусти оцінку", "зроби оцінку",
            "run evaluation", "evaluation",
            "eval", "оцінка якості"
        ],
        "handler": _eval_from_text,
    },
    {
        "name": "memory",
        "patterns": ["покажи пам", "memory", "пам’ять", "память"],
        "handler": lambda text: show_memory(),
    },
    {
        "name": "agents",
        "patterns": [
            "покажи список агентів",
            "покажи агентів",
            "список агентів",
            "agents list",
            "show agents"
        ],
        "handler": lambda text: show_agents(),
    },
    {
        "name": "errors",
        "patterns": [
            "покажи помилки", "покажи помилкі",
            "помилки", "помилкі",
            "errors", "проблеми", "issues"
        ],
        "handler": _errors_from_text,
    },
    {
        "name": "runs",
        "patterns": [
            "покажи останні запуски", "покажи остані запуски",
            "останні запуски", "остані запуски",
            "last runs", "історія запусків", "history",
            "покажи запуск", "покажи запуски"
        ],
        "handler": _runs_from_text,
    },
    {
        "name": "db_runs",
        "patterns": [
            "покажи запуски з бд",
            "запуски з бд",
            "бд запуски",
            "db runs",
            "runs from db",
        ],
        "handler": _db_runs_from_text,
    },
    {
        "name": "dataset_add",
        "patterns": [
            "додай запуск в датасет",
            "додай в датасет",
            "add to dataset",
            "dataset add",
            "mark run",
            "mark to dataset",
        ],
        "handler": _dataset_add_from_text,
    },
    {
        "name": "dataset_show",
        "patterns": [
            "покажи датасет",
            "dataset",
            "dataset show",
            "приклади датасету",
        ],
        "handler": _dataset_show_from_text,
    },
    {
        "name": "projects",
        "patterns": [
            "список проєктів",
            "список проектів",
            "проєкти",
            "проекти",
            "projects list",
            "show projects",
        ],
        "handler": _projects_from_text,
    },
    {
        "name": "current_project",
        "patterns": [
            "поточний проєкт",
            "поточний проект",
            "current project",
            "який зараз проєкт",
            "який зараз проект",
        ],
        "handler": _current_project_from_text,
    },
    {
        "name": "meta_train",
        "patterns": [
            "meta тренування",
            "мета тренування",
            "meta training",
            "meta train",
            "запусти meta тренування",
        ],
        "handler": _meta_train_from_text,
    },
    {
        "name": "trainer_analysis",
        "patterns": [
            "аналіз агентів",
            "аналіз запусків",
            "trainer stats",
            "аналіз бд запусків",
            "аналіз останніх запусків",
        ],
        "handler": _trainer_from_text,
    },
    {
        "name": "trainer_apply_configs",
        "patterns": [
            "застосуй пропозиції тренера",
            "онови конфіг агента за тренером",
            "онови конфіги агентів за тренером",
            "apply trainer suggestions",
            "trainer apply configs",
        ],
        "handler": _trainer_apply_from_text,
    },
]


def match_command(text: str) -> Optional[Callable[[], str]]:
    t = text.lower().strip()

    # smart fallback for current project when користувач пише просто "проєкт"/"проект"/"project"
    if t in ("проєкт", "проект", "project"):
        return lambda x=text: _current_project_from_text(x)

    # smart fallback for "runs" with declensions/typos
    # вимагає наявності слова "покажи" або "show", щоб не перехоплювати фрази типу "аналіз 30 останніх запусків"
    if ("покажи" in t or "show" in t) and re.search(r"\bостанн\w*\b", t) and re.search(r"\bзапуск\w*\b", t):
        return lambda x=text: _runs_from_text(x)

    # smart fallback for "errors" with day hint
    if re.search(r"\bпомилк\w*\b", t) and ("сьогодні" in t or "вчора" in t):
        return lambda x=text: _errors_from_text(x)

    # smart fallback for adding to dataset
    if "датасет" in t and ("додай" in t or "add" in t or "mark" in t):
        return lambda x=text: _dataset_add_from_text(x)

    # smart fallback for showing dataset
    if "датасет" in t and ("покажи" in t or "show" in t):
        return lambda x=text: _dataset_show_from_text(x)

    # smart fallback for trainer analysis (аналіз запусків / агентів)
    if "аналіз" in t and ("запуск" in t or "запусків" in t or "бд" in t or "агент" in t or "agents" in t):
        return lambda x=text: _trainer_from_text(x)

    for cmd in COMMANDS:
        for p in cmd["patterns"]:
            if p in t:
                handler = cmd["handler"]
                return lambda h=handler, x=text: h(x)

    return None
