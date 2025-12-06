import argparse

from agents.supervisor import Supervisor
from memory.store import load_memory, save_memory, set_flag


def learn_from_tags(memory, tags):
    if "structure" in tags:
        set_flag(memory, "force_structure", True)
    if "too_short" in tags:
        set_flag(memory, "expand_when_short", True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learn", action="store_true", help="Update memory flags after each task.")
    args = parser.parse_args()

    memory = load_memory()
    sup = Supervisor()

    print("Multi-agent chat. Type 'exit' to quit.")

    while True:
        task = input("\nYou: ").strip()
        if not task:
            continue
        if task.lower() in {"exit", "quit"}:
            break

        result = sup.run(task, memory)
        print("\nAssistant:\n")
        print(result["final"])

        if args.learn:
            tags = result.get("critique_tags", []) or []
            learn_from_tags(memory, tags)
            save_memory(memory)


if __name__ == "__main__":
    main()
