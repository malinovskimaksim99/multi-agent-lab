import argparse
import json
from datetime import datetime
from pathlib import Path

from agents.supervisor import Supervisor
from memory.store import load_memory, save_memory, set_flag

LOGS = Path("logs.jsonl")

def log_run(result):
    if not LOGS.exists():
        LOGS.write_text("", encoding="utf-8")
    with LOGS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            **result
        }, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="Explain what a multi-agent system is in 5 bullet points."
    )
    parser.add_argument(
        "--learn",
        action="store_true",
        help="Enable simple flag learning after critique."
    )
    args = parser.parse_args()

    memory = load_memory()
    sup = Supervisor()

    result = sup.run(args.task, memory)
    print(result["final"])

    if args.learn:
        tags = result.get("critique_tags", [])
        if "structure" in tags:
            set_flag(memory, "force_structure", True)
            save_memory(memory)

    log_run(result)

if __name__ == "__main__":
    main()
