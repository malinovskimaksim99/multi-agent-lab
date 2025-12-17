import argparse
import json

import agents  # noqa: F401
from agents.router import rank_agents
from agents.supervisor import Supervisor
from agents.registry import list_agents
from memory.store import load_memory, save_memory, set_flag

from commands import match_command


def learn_from_tags(memory, tags):
    if "structure" in tags:
        set_flag(memory, "force_structure", True)
    if "too_short" in tags:
        set_flag(memory, "expand_when_short", True)


class HeadAgent:
    """
    Простий локальний "головний агент" для чат-режиму.

    Він:
    - приймає вхідний текст користувача;
    - обробляє службові /memory, /agents, /route;
    - пробує знайти natural-language команду через match_command();
    - якщо це не команда, делегує задачу Supervisor-у.
    """

    def __init__(self, sup: Supervisor, memory: dict, args: argparse.Namespace):
        self.sup = sup
        self.memory = memory
        self.args = args

    def handle(self, task: str) -> bool:
        """
        Обробляє один запит користувача.
        Повертає False, якщо потрібно завершити чат (exit/quit), інакше True.
        """
        if not task:
            return True

        if task.lower() in {"exit", "quit"}:
            return False

        # --- slash service commands ---
        if task in {"/memory", ":memory"}:
            print(json.dumps(load_memory(), ensure_ascii=False, indent=2))
            return True

        if task in {"/agents", ":agents"}:
            print("Registered agents:", ", ".join(list_agents()))
            return True

        if task.startswith("/route"):
            query = task[len("/route"):].strip()
            if not query:
                print("Usage: /route <task>")
                return True

            ranked = rank_agents(query, self.memory)
            print("Routing scores:")
            for name, score in ranked:
                print(f" - {name}: {score:.2f}")
            return True

        # --- natural language commands ---
        handler = match_command(task)
        if handler:
            out = handler()
            print("\nAssistant:\n")
            print(out)
            return True

        # --- normal agent task (через Supervisor) ---
        result = self.sup.run(task, self.memory)

        solver_name = result.get("solver_agent", "unknown")
        team_agents = result.get("team_agents")

        header = f"solver: {solver_name}"
        if team_agents:
            header += f" | team: {', '.join(team_agents)}"

        print(f"\nAssistant ({header}):\n")
        print(result["final"])

        if self.args.learn:
            tags = result.get("critique_tags", []) or []
            learn_from_tags(self.memory, tags)
            save_memory(self.memory)

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learn", action="store_true", help="Update memory flags after each task.")
    parser.add_argument("--auto", action="store_true", help="Auto-pick best single solver agent by router.")
    parser.add_argument("--team", action="store_true", help="Use team mode (top-N solvers + synthesizer).")
    parser.add_argument("--team-size", type=int, default=2, help="Team size for --team mode.")
    args = parser.parse_args()

    memory = load_memory()
    sup = Supervisor(
        auto_solver=args.auto and not args.team,
        auto_team=args.team,
        team_size=args.team_size,
    )

    head = HeadAgent(sup, memory, args)

    print("Multi-agent chat. Type 'exit' to quit.")
    print("Commands: /memory, /agents, /route <task> + natural commands like 'зроби звіт'")

    while True:
        task = input("\nYou: ").strip()
        should_continue = head.handle(task)
        if not should_continue:
            break


if __name__ == "__main__":
    main()
