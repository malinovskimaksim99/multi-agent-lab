import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agents.supervisor import Supervisor
from memory.store import load_memory, save_memory, set_flag

LOGS = Path("logs.jsonl")


def log_run(result):
    if not LOGS.exists():
        LOGS.write_text("", encoding="utf-8")
    with LOGS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            **result
        }, ensure_ascii=False) + "\n")


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
    args = parser.parse_args()

    memory = load_memory()
    sup = Supervisor(auto_solver=args.auto)

    result = sup.run(args.task, memory)

    solver = result.get("solver_agent")
    tags = result.get("critique_tags")
    if solver or tags:
        print(f"[solver: {solver} | tags: {tags}]")

    print(result["final"])

    if args.learn:
        tags = result.get("critique_tags", []) or []
        learn_from_tags(memory, tags)
        save_memory(memory)

    log_run(result)


if __name__ == "__main__":
    main()
