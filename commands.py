import json
import sys
import subprocess
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

LOGS = Path("logs.jsonl")


def _safe_read_logs(limit: int = 50) -> List[Dict[str, Any]]:
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


def show_memory() -> str:
    from memory.store import load_memory
    return json.dumps(load_memory(), ensure_ascii=False, indent=2)


def show_agents() -> str:
    from agents.registry import list_agents
    agents_list = list_agents()
    return "Registered agents: " + ", ".join(agents_list)


def show_recent_runs(limit: int = 10) -> str:
    data = _safe_read_logs(limit=200)
    if not data:
        return "Немає логів запусків поки що."
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


def show_errors(limit: int = 20) -> str:
    data = _safe_read_logs(limit=300)
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


def help_text() -> str:
    return (
        "Доступні живі команди (можна писати звичайним текстом):\n"
        "- зроби звіт / progress report\n"
        "- покажи пам’ять / memory\n"
        "- покажи помилки / errors\n"
        "- покажи останні запуски / last runs\n"
        "- покажи агентів / список агентів\n"
        "\nТакож залишаються slash-команди:\n"
        "- /memory, /agents, /route <task>"
    )


COMMANDS: List[Dict[str, Any]] = [
    {
        "name": "help",
        "patterns": ["help", "довідка", "команди", "що вмієш", "що можеш"],
        "handler": help_text,
    },
    {
        "name": "report",
        "patterns": ["зроби звіт", "покажи звіт", "progress report", "прогрес"],
        "handler": run_progress_report,
    },
    {
        "name": "memory",
        "patterns": ["покажи пам", "memory", "пам’ять", "память"],
        "handler": show_memory,
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
        "handler": show_agents,
    },
    {
        "name": "errors",
        "patterns": [
            "покажи помилки", "покажи помилкі",
            "помилки", "помилкі",
            "errors", "проблеми", "issues"
        ],
        "handler": show_errors,
    },
    {
        "name": "runs",
        "patterns": [
            "покажи останні запуски", "покажи остані запуски",
            "останні запуски", "остані запуски",
            "last runs", "історія запусків", "history"
        ],
        "handler": show_recent_runs,
    },
]


def match_command(text: str) -> Optional[Callable[[], str]]:
    t = text.lower().strip()
    for cmd in COMMANDS:
        for p in cmd["patterns"]:
            if p in t:
                return cmd["handler"]
    return None
