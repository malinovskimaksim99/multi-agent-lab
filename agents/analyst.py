from typing import Optional, List, Dict, Any

from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent
from db import get_agent_config


def _is_docs(t: str) -> bool:
    return any(k in t for k in ["readme", "installation", "docs", "documentation", "guide"])


def _is_explain(t: str) -> bool:
    markers = [
        "explain",
        "difference",
        "compare",
        "roles of",
        "summary",
        "summarize",
        "overview",
        "vs",
        "versus",
    ]
    ua = ["поясни", "різниц", "порівняй", "що таке", "огляд", "підсумуй"]
    return any(m in t for m in markers) or any(m in t for m in ua)


def _is_planning(t: str) -> bool:
    markers = ["plan", "planning", "roadmap", "outline", "next steps", "strategy"]
    ua = ["план", "кроки", "наступні кроки", "дорожня карта", "стратег"]
    return any(m in t for m in markers) or any(m in t for m in ua)


@register_agent
class AnalystAgent(BaseAgent):
    name = "analyst"
    description = "General problem solver focused on clarity, structure, and task-specific reasoning."

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        t = task.lower()
        if _is_docs(t):
            return 0.55  # writer має бути головним по документації
        if _is_explain(t):
            return 0.70  # для explain зазвичай буде спеціальний explainer
        if _is_planning(t):
            return 0.85
        return 0.75

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        t = task.lower().strip()

        # легкий прапорець структури з memory + конфіг з БД
        flags = memory.get("flags") or {}
        cfg: Dict[str, Any] = {}

        try:
            fs = get_agent_config(self.name, "force_structure")
        except Exception:
            fs = None

        try:
            mp = get_agent_config(self.name, "max_points")
        except Exception:
            mp = None

        if fs is not None:
            cfg["force_structure"] = fs
        if mp is not None:
            cfg["max_points"] = mp

        force_structure = bool(
            flags.get("force_structure", False) or cfg.get("force_structure", False)
        )
        max_points = cfg.get("max_points")

        # --- 1) Planning vs critique ---
        if (
            "planning vs critique" in t
            or (
                ("planning" in t or "планув" in t or "план" in t)
                and ("critique" in t or "критик" in t)
                and ("vs" in t or "різниц" in t or "difference" in t or "compare" in t)
            )
        ):
            output = (
                "## Планування vs критика\n\n"
                "**Планування** — це про те, що і в якій послідовності робити. "
                "Воно визначає цілі, кроки та ресурси.\n\n"
                "**Критика** — це про оцінку вже зробленої роботи. "
                "Вона допомагає знайти помилки, прогалини та місця, які можна покращити.\n\n"
                "### Планування\n"
                "- Формулює ціль і очікуваний результат.\n"
                "- Розбиває задачу на кроки та підзадачі.\n"
                "- Допомагає обрати потрібних агентів або інструменти.\n\n"
                "### Критика\n"
                "- Читає чернетку або проміжний результат.\n"
                "- Виявляє логічні помилки, нестачу прикладів чи кроків.\n"
                "- Дає конкретні зауваження та рекомендації.\n\n"
                "### Як вони працюють разом\n"
                "1. Планування: визначаємо, що треба зробити.\n"
                "2. Виконання: аналітик або інший агент створює чернетку.\n"
                "3. Критика: оцінюємо результат, шукаємо покращення.\n"
                "4. Оновлений план / правки: оновлюємо підхід і робимо кращу версію.\n\n"
                "**Висновок:** планування задає напрямок і структуру роботи, а критика допомагає довести рішення до якісного стану."
            )
            return AgentResult(
                agent=self.name,
                output=output,
                meta={"mode": "answer_planning_vs_critique"},
            )

        # --- 2) Ролі Planner / Analyst / Critic ---
        if (
            ("roles of" in t or "role" in t or "роль" in t or "ролі" in t)
            and (
                "planner" in t
                or "analyst" in t
                or "critic" in t
                or "планувальник" in t
                or "аналітик" in t
                or "критик" in t
            )
        ):
            output = (
                "## Ролі Planner, Analyst і Critic\n\n"
                "### Planner (планувальник)\n"
                "- Уточнює задачу і очікуваний результат.\n"
                "- Формує план із основних кроків.\n"
                "- Допомагає обрати, яких агентів варто залучити.\n\n"
                "### Analyst (аналітик)\n"
                "- Детально розбирає задачу.\n"
                "- Створює основну відповідь: текст, код, план дій.\n"
                "- Пояснює, чому обрав саме такий підхід.\n\n"
                "### Critic (критик)\n"
                "- Перевіряє чернетку відповіді.\n"
                "- Знаходить прогалини, нечіткі місця, помилки.\n"
                "- Виставляє теги якості та формує список зауважень.\n\n"
                "### Як вони взаємодіють\n"
                "1. Planner формує план.\n"
                "2. Analyst виконує роботу за цим планом.\n"
                "3. Critic оцінює результат і підказує, як його покращити.\n"
                "4. За потреби цикл повторюється до якісної версії.\n"
            )
            return AgentResult(
                agent=self.name,
                output=output,
                meta={"mode": "answer_roles"},
            )

        # --- 3) Порівняння single-solver vs team-solver ---
        if (
            ("single-solver" in t or "single solver" in t or "одиночн" in t)
            and ("team" in t or "командн" in t)
            and ("vs" in t or "difference" in t or "compare" in t or "різниц" in t)
        ):
            output = (
                "## Single-solver vs team-solver режими\n\n"
                "### Single-solver режим\n"
                "- Задача виконується одним основним агентом (наприклад, Analyst або Coder).\n"
                "- Простіше та швидше для нескладних запитів.\n"
                "- Менше координації, але й менше різних точок зору.\n\n"
                "### Team-solver режим\n"
                "- Кілька агентів працюють над задачею паралельно.\n"
                "- Кожен приносить свою спеціалізацію (документація, код, пояснення тощо).\n"
                "- Потрібен Synthesizer, щоб обʼєднати результати в одну відповідь.\n\n"
                "### Коли який режим краще\n"
                "- Single-solver: для простих, локальних питань без великої кількості деталей.\n"
                "- Team-solver: для комплексних задач, де важливі різні аспекти (пояснення, код, архітектура, документація).\n"
            )
            return AgentResult(
                agent=self.name,
                output=output,
                meta={"mode": "answer_single_vs_team"},
            )

        # --- 4) Опис multi-agent pipeline ---
        if (
            ("multi-agent" in t or "multi agent" in t or "мультиагент" in t)
            and ("pipeline" in t or "пайплайн" in t or "pipeline works" in t or "як працює" in t)
        ):
            output = (
                "## Як працює multi-agent pipeline\n\n"
                "1. Користувач формулює запит.\n"
                "2. Planner переформульовує його в чітке завдання та план.\n"
                "3. Supervisor/Router обирають відповідних агентів (один або команда).\n"
                "4. Обрані агенти виконують свою частину роботи.\n"
                "5. Critic перевіряє чернетку, додає нотатки й теги якості.\n"
                "6. Synthesizer обʼєднує результати в фінальну відповідь.\n"
                "7. Памʼять або логи зберігають інформацію для майбутніх запусків.\n"
            )
            return AgentResult(
                agent=self.name,
                output=output,
                meta={"mode": "answer_pipeline"},
            )

        # --- Загальні скелети відповіді, але вже українською ---

        lines: List[str] = []

        if _is_docs(t):
            lines = [
                "Коротко опишіть призначення цього розділу.",
                "Перерахуйте мінімальні кроки встановлення (клон, залежності, запуск).",
                "Додайте один рядок про перевірку, що все працює.",
                "За потреби згадайте специфіку операційної системи.",
            ]
        elif _is_planning(t):
            lines = [
                "Уточніть ціль і критерії успіху.",
                "Перерахуйте 3–5 конкретних кроків з послідовністю.",
                "Додайте короткий крок перевірки результату.",
                "Зазначте можливі ризики або типові помилки.",
            ]
        elif _is_explain(t):
            lines = [
                "Дайте коротке визначення основних термінів.",
                "Поясніть, для чого це використовується на практиці.",
                "Порівняйте з найближчими схожими поняттями, якщо це доречно.",
                "Наведіть простий приклад.",
                "Зробіть короткий висновок (1–2 речення).",
            ]
        else:
            lines = [
                "Сформулюйте задачу своїми словами.",
                "Дайте 3–5 конкретних пунктів, що напряму відповідають на питання.",
                "За потреби додайте короткий приклад або пораду.",
            ]

        # Якщо в конфізі задано max_points — обрізаємо кількість пунктів
        if isinstance(max_points, int) and max_points > 0 and len(lines) > max_points:
            lines = lines[:max_points]

        if force_structure:
            out = (
                "## Завдання\n"
                f"{task}\n\n"
                "## Ключові пункти\n"
                + "\n".join(f"- {l}" for l in lines)
            )
        else:
            out = "\n".join(f"- {l}" for l in lines)

        return AgentResult(
            agent=self.name,
            output=out,
            meta={
                "mode": "content_scaffold_uk",
                "config": cfg,
            },
        )
