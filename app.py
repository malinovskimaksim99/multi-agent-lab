import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agents.supervisor import Supervisor
from memory.store import load_memory, save_memory, set_flag
from db import init_db, save_run_to_db

LOGS = Path("logs.jsonl")


def log_run(result):
    """Логує запуск у logs.jsonl та записує його в SQLite-базу."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **result,
    }

    # Лог у файл
    if not LOGS.exists():
        LOGS.write_text("", encoding="utf-8")
    with LOGS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Запис у БД
    try:
        save_run_to_db(entry)
    except Exception as e:
        # Не валимо основний сценарій, якщо з БД щось не так
        print(f"[warn] Не вдалося зберегти запуск у БД: {e}")


def learn_from_tags(memory, tags):
    if "structure" in tags:
        set_flag(memory, "force_structure", True)
    if "too_short" in tags:
        set_flag(memory, "expand_when_short", True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--team", action="store_true")
    parser.add_argument("--team-size", type=int, default=2)
    args = parser.parse_args()

    # Ініціалізуємо БД (створює таблиці, якщо їх ще немає)
    init_db()

    memory = load_memory()
    sup = Supervisor(
        auto_solver=args.auto and not args.team,
        auto_team=args.team,
        team_size=args.team_size,
    )

    result = sup.run(args.task, memory)

    solver = result.get("solver_agent")
    team = result.get("team_agents")
    tags = result.get("critique_tags")

    line = f"[solver: {solver} | tags: {tags}]"
    if team:
        line = f"[solver: {solver} | team: {team} | tags: {tags}]"
    print(line)

    print(result["final"])

    if args.learn:
        crit_tags = result.get("critique_tags", []) or []
        learn_from_tags(memory, crit_tags)
        save_memory(memory)

    log_run(result)


if __name__ == "__main__":
    main()
