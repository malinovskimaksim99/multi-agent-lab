import json
from datetime import datetime
from pathlib import Path

from agents.supervisor import Supervisor
from memory.store import load_memory, save_memory, add_rule

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
    task = "Analyze this short text and provide 3 key insights, then check for logical gaps."
    memory = load_memory()

    sup = Supervisor()
    result = sup.run(task, memory)
    print(result["final"])

    # simple "learning"
    if "structure" in result["critique"].lower():
        add_rule(memory, "Always format the answer with clear headings/bullets.")
        save_memory(memory)

    log_run(result)

if __name__ == "__main__":
    main()
