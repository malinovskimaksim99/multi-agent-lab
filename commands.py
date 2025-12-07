import json
import sys
import subprocess
import re
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime, timezone, timedelta

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


def help_text() -> str:
    return (
        "Доступні живі команди:\n"
        "- зроби звіт / progress report\n"
        "- запусти оцінку / eval\n"
        "- покажи пам’ять / memory\n"
        "- покажи агентів / список агентів\n"
        "- покажи помилки / errors\n"
        "- покажи останні запуски / last runs\n"
        "\nПараметризовані приклади:\n"
        "- покажи останні 20 запусків\n"
        "- покажи помилки за сьогодні\n"
        "- покажи останні 5 запусків за вчора\n"
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
]


def match_command(text: str) -> Optional[Callable[[], str]]:
    t = text.lower().strip()

    # smart fallback for "runs" with declensions/typos
    if re.search(r"\bостанн\w*\b", t) and re.search(r"\bзапуск\w*\b", t):
        return lambda x=text: _runs_from_text(x)

    # smart fallback for "errors" with day hint
    if re.search(r"\bпомилк\w*\b", t) and ("сьогодні" in t or "вчора" in t):
        return lambda x=text: _errors_from_text(x)

    for cmd in COMMANDS:
        for p in cmd["patterns"]:
            if p in t:
                handler = cmd["handler"]
                return lambda h=handler, x=text: h(x)

    return None
