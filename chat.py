import argparse
import json

import agents  # тригерить agents/__init__.py і реєстрацію плагінів
from agents.router import rank_agents
from agents.supervisor import Supervisor
from agents.registry import list_agents
from memory.store import load_memory, save_memory, set_flag


def learn_from_tags(memory, tags):
    if "structure" in tags:
        set_flag(memory, "force_structure", True)
    if "too_short" in tags:
        set_flag(memory, "expand_when_short", True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learn", action="store_true", help="Update memory flags after each task.")
    parser.add_argument("--auto", action="store_true", help="Auto-pick best solver agent by router.")
    args = parser.parse_args()

    memory = load_memory()
    sup = Supervisor(auto_solver=args.auto)

    print("Multi-agent chat. Type 'exit' to quit.")
    print("Commands: /memory, /agents, /route <task>")

    while True:
        task = input("\nYou: ").strip()
        if not task:
            continue

        if task.lower() in {"exit", "quit"}:
            break

        # --- service commands ---
        if task in {"/memory", ":memory"}:
            print(json.dumps(load_memory(), ensure_ascii=False, indent=2))
            continue

        if task in {"/agents", ":agents"}:
            print("Registered agents:", ", ".join(list_agents()))
            continue

        # --- routing preview ---
        if task.startswith("/route"):
            query = task[len("/route"):].strip()
            if not query:
                print("Usage: /route <task>")
                continue

            ranked = rank_agents(query, memory)
            print("Routing scores:")
            for name, score in ranked:
                print(f" - {name}: {score:.2f}")
            continue

        # --- normal agent task ---
        result = sup.run(task, memory)
        solver_name = result.get("solver_agent", "unknown")

        print(f"\nAssistant (solver: {solver_name}):\n")
        print(result["final"])

        if args.learn:
            tags = result.get("critique_tags", []) or []
            learn_from_tags(memory, tags)
            save_memory(memory)


if __name__ == "__main__":
    main()
