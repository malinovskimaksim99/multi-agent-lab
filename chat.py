import argparse
import json

import agents  # noqa: F401
from agents.head import HeadAgent as CoreHeadAgent
from agents.registry import list_agents
from agents.router import rank_agents
from agents.supervisor import Supervisor
from commands import match_command
from memory.store import load_memory, save_memory


class CliChat:
    """CLI обгортка для HeadAgent.

    Важливо:
    - /memory, /agents, /route лишаємо для дебагу
    - всі звичайні повідомлення йдуть через CoreHeadAgent (LLM + авто-делегування)
    """

    def __init__(self, head: CoreHeadAgent, sup: Supervisor, memory: dict, args: argparse.Namespace):
        self.head = head
        self.sup = sup
        self.memory = memory
        self.args = args

        # Підміняємо Supervisor усередині HeadAgent, щоб CLI-параметри (--auto/--team) працювали
        self.head.supervisor = sup

    def handle(self, task: str) -> bool:
        """Обробляє один запит користувача.

        Повертає False, якщо треба завершити чат (exit/quit), інакше True.
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
            query = task[len("/route") :].strip()
            if not query:
                print("Usage: /route <task>")
                return True

            ranked = rank_agents(query, self.memory)
            print("Routing scores:")
            for name, score in ranked:
                print(f" - {name}: {score:.2f}")
            return True

        # --- natural language commands (локальні) ---
        handler = match_command(task)
        if handler:
            out = handler()
            print("\nAssistant:\n")
            print(out)
            return True

        # --- main path: HeadAgent ---
        reply = self.head.handle(task, self.memory, auto=True)

        print("\nAssistant:\n")
        print(reply)

        # Навіть якщо зараз ми не вчимося з critique_tags у цьому режимі,
        # ми все одно зберігаємо пам'ять (якщо користувач запускає з --learn).
        if self.args.learn:
            save_memory(self.memory)

        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learn", action="store_true", help="Save memory after each message.")
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

    head = CoreHeadAgent()
    chat = CliChat(head, sup, memory, args)

    print("HeadAgent chat. Type 'exit' to quit.")
    print("Commands: /memory, /agents, /route <task> + natural commands like 'зроби звіт'")

    while True:
        task = input("\nYou: ").strip()
        if not chat.handle(task):
            break


if __name__ == "__main__":
    main()
