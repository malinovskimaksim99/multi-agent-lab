import json
from collections import Counter

from agents.supervisor import Supervisor
from memory.store import load_memory, save_memory, add_rule

# Примітивні тригери -> правила пам’яті
KEYWORD_TO_RULE = {
    "structure": "Always format the answer with clear headings or bullet points.",
    "too short": "When giving guidance, include at least 3 concrete steps or examples."
}

def main():
    with open("tests/sample_tasks.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)["tasks"]

    memory = load_memory()
    sup = Supervisor()
    stats = Counter()

    for task in tasks:
        result = sup.run(task, memory)
        critique = (result.get("critique") or "").lower()

        # збираємо статистику і додаємо правила
        for ключ, правило in KEYWORD_TO_RULE.items():
            if ключ in critique:
                stats[ключ] += 1
                add_rule(memory, правило)

    save_memory(memory)

    print("Test run complete.")
    print("Critique keyword stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nCurrent memory rules:")
    for r in memory.get("rules", []):
        print(f" - {r}")

if __name__ == "__main__":
    main()
