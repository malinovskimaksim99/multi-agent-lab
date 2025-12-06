import json
from collections import Counter

from agents.supervisor import Supervisor
from memory.store import load_memory, save_memory, set_flag

def main():
    with open("tests/sample_tasks.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)["tasks"]

    memory = load_memory()
    sup = Supervisor()
    stats = Counter()

    for task in tasks:
        result = sup.run(task, memory)
        tags = result.get("critique_tags", [])

        for t in tags:
            stats[t] += 1

        # learning rules (very simple)
        if "structure" in tags:
            set_flag(memory, "force_structure", True)

    save_memory(memory)

    print("Test run complete.")
    print("Critique tag stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nMemory flags now:")
    print(memory.get("flags", {}))

if __name__ == "__main__":
    main()
