from typing import Optional, List, Dict, Any
from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent
from db import get_agent_config


def _load_agent_config(agent_name: str) -> Dict[str, Any]:
    """
    Завантажує конфіг агента з БД через get_agent_config.
    Повертає словник з ключами типу 'style', 'max_lines' тощо.
    """
    cfg: Dict[str, Any] = {}
    try:
        style = get_agent_config(agent_name, "style")
        if style:
            cfg["style"] = style
    except Exception:
        pass

    try:
        max_lines = get_agent_config(agent_name, "max_lines")
        if max_lines not in (None, ""):
            try:
                cfg["max_lines"] = int(max_lines)
            except Exception:
                pass
    except Exception:
        pass

    return cfg


def _apply_style_to_text(text: str, config: Dict[str, Any]) -> str:
    """
    Застосовує стиль форматування до готового тексту пояснення:
      - style="compact" прибирає зайві порожні рядки;
      - max_lines скорочує відповідь до N рядків, якщо потрібно.
    """
    if not text:
        return text

    style = config.get("style")
    max_lines = config.get("max_lines")

    lines = text.splitlines()

    # Обрізаємо по кількості рядків, якщо задано
    if isinstance(max_lines, int) and max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]

    # Компактний стиль: прибираємо дублікати порожніх рядків
    if style == "compact":
        cleaned: List[str] = []
        prev_blank = False
        for ln in lines:
            s = ln.rstrip()
            is_blank = (s == "")
            if is_blank and prev_blank:
                continue
            cleaned.append(s)
            prev_blank = is_blank
        lines = cleaned

    return "\n".join(lines)


@register_agent
class ExplainerAgent(BaseAgent):
    name = "explainer"
    description = "Specialist for clear explanations, differences, and how/why questions."

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        """
        Оцінює, наскільки цей агент підходить для задачі пояснення.

        Ми комбінуємо:
        - маркери в самому тексті запиту (англ/укр),
        - можливий task_type з context (якщо його вже визначив Supervisor/Router).
        """
        t = task.lower()
        ctx: Context = context or {}
        task_type = ctx.get("task_type")

        # 0) Дуже сильні спеціальні кейси, де explainer має бути головним:
        is_planning_vs_critique = (
            "planning vs critique" in t
            or (
                ("planning" in t or "планув" in t or "план" in t)
                and ("critique" in t or "критик" in t)
            )
        )

        role_syns = ["roles of", "role", "роль", "ролі"]
        agent_syns = ["planner", "планувальник", "аналітик", "analyst", "критик", "critic"]
        is_roles_question = (
            any(r in t for r in role_syns)
            and any(a in t for a in agent_syns)
        )

        single_syns = ["single-solver", "single solver", "одиночн"]
        team_syns = ["team-solver", "team solver", "team", "командн"]
        compare_syns = ["vs", "versus", "difference", "compare", "різниц"]
        is_single_vs_team = (
            any(s in t for s in single_syns)
            and any(tm in t for tm in team_syns)
            and any(c in t for c in compare_syns)
        )

        if is_planning_vs_critique or is_roles_question or is_single_vs_team:
            # Для цих задач explainer повинен впевнено вигравати у analyst
            score = 1.30
        else:
            # 1) Маркери в тексті запиту
            strong_en = [
                "explain", "difference", "compare", "why", "how",
                "what is", "roles of", "versus", "vs"
            ]
            strong_ua = [
                "поясни", "різниц", "порівняй", "чому", "як працює",
                "що таке", "огляд", "пояснення"
            ]

            if any(s in t for s in strong_en) or any(s in t for s in strong_ua):
                score = 0.92
            else:
                medium_en = ["meaning", "concept", "overview", "summary", "introduce"]
                medium_ua = ["значен", "понятт", "короткий огляд", "підсумок"]
                if any(m in t for m in medium_en) or any(m in t for m in medium_ua):
                    score = 0.65
                else:
                    score = 0.15

        # 2) Підлаштування під task_type з контексту
        if task_type == "explain":
            # Якщо роутер класифікував задачу як explain, підсилюємо explainer
            score = max(score, 0.9)
        elif task_type == "plan":
            # Для чистого планування Analyst зазвичай кращий, не агресивно піднімаємо
            score = max(score, 0.4)
        elif task_type == "docs":
            # Документацію зазвичай пише writer, але explainer теж може допомогти
            score = max(score, 0.5)

        return score

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        flags = memory.get("flags", {}) or {}
        force_structure = bool(flags.get("force_structure"))
        expand_when_short = bool(flags.get("expand_when_short"))

        t = task.lower().strip()

        # Завантажуємо конфіг для explainer з БД
        config = _load_agent_config(self.name)

        output: str
        mode: str

        # --- Special case 1: planning vs critique ---
        planning_syns = ["planning", "планув", "план"]
        critique_syns = ["critique", "критик"]
        if (
            "planning vs critique" in t
            or (
                any(p in t for p in planning_syns)
                and any(c in t for c in critique_syns)
            )
        ):
            output = (
                "## Планування vs критика\n\n"
                "**Планування** — це процес визначення цілей та кроків для досягнення результату. "
                "**Критика** — це оцінка виконаної роботи з метою виявлення помилок і покращення.\n\n"
                "### Планування\n"
                "- Формулює мету та бажаний результат.\n"
                "- Визначає послідовність кроків.\n"
                "- Обирає інструменти та ресурси.\n"
                "- Допомагає уникати зайвих дій.\n\n"
                "### Критика\n"
                "- Оцінює отриманий результат або чернетку.\n"
                "- Допомагає знайти помилки чи недоліки.\n"
                "- Дає конструктивні зауваження для покращення.\n"
                "- Виставляє теги якості або рекомендації.\n\n"
                "### Як вони працюють разом\n"
                "1. Планування: визначаємо цілі та кроки.\n"
                "2. Виконання: реалізуємо задумане.\n"
                "3. Критика: аналізуємо результат, шукаємо покращення.\n"
                "4. Вносимо зміни та вдосконалюємо рішення.\n\n"
                "**Висновок:** Планування задає напрямок руху, а критика допомагає зробити результат якіснішим та уникнути типових помилок."
            )
            mode = "explain_planning_vs_critique"

        # --- Special case 2: Roles of Planner / Analyst / Critic ---
        else:
            role_syns = ["roles of", "role", "роль", "ролі"]
            agent_syns = ["planner", "планувальник", "аналітик", "analyst", "критик", "critic"]
            if (
                any(r in t for r in role_syns)
                and any(a in t for a in agent_syns)
            ):
                output = (
                    "## Ролі Planner, Analyst і Critic\n\n"
                    "### Planner (планувальник)\n"
                    "- Уточнює суть задачі та вимоги.\n"
                    "- Формує чітку ціль.\n"
                    "- Розбиває задачу на кроки або підзадачі.\n"
                    "- Пропонує відповідних агентів або інструменти.\n\n"
                    "### Analyst (аналітик)\n"
                    "- Аналізує поставлену задачу.\n"
                    "- Виконує основну частину роботи: генерує відповідь, рішення чи код.\n"
                    "- Пояснює логіку та кроки виконання.\n"
                    "- Дотримується плану та враховує умови.\n\n"
                    "### Critic (критик)\n"
                    "- Перевіряє чернетку або готову відповідь.\n"
                    "- Виявляє помилки, неясності чи прогалини.\n"
                    "- Виставляє теги якості (наприклад, «логічність», «повнота»).\n"
                    "- Складає список зауважень або рекомендацій.\n\n"
                    "### Як вони взаємодіють\n"
                    "Планувальник визначає шлях, аналітик виконує роботу, критик оцінює і допомагає покращити результат. Цикл повторюється до досягнення якісної відповіді."
                )
                mode = "explain_roles"

            # --- Special case 3: Single-solver vs team-solver comparison ---
            else:
                single_syns = ["single-solver", "single solver", "одиночн"]
                team_syns = ["team-solver", "team solver", "team", "командн"]
                compare_syns = ["vs", "versus", "difference", "compare", "різниц"]

                if (
                    any(s in t for s in single_syns)
                    and any(tm in t for tm in team_syns)
                    and any(c in t for c in compare_syns)
                ):
                    output = (
                        "## Single-solver vs team-solver режими\n\n"
                        "### Single-solver\n"
                        "- Задача виконується одним основним агентом (наприклад, Analyst або Coder).\n"
                        "- Підходить для простих, локальних запитів.\n"
                        "- Швидший запуск і менше координації.\n"
                        "- Менше різних точок зору, але й менше дублювання роботи.\n\n"
                        "### Team-solver\n"
                        "- Над задачею працює кілька агентів паралельно.\n"
                        "- Кожен агент відповідає за свою спеціалізацію (код, пояснення, документація тощо).\n"
                        "- Потрібен Synthesizer, щоб обʼєднати відповіді в один узгоджений результат.\n"
                        "- Краще підходить для складних задач з багатьма аспектами.\n\n"
                        "### Коли який режим кращий\n"
                        "- Обирайте single-solver, коли задача проста і не потребує кількох експертів.\n"
                        "- Обирайте team-solver, коли потрібно поєднати різні компетенції (архітектура, код, пояснення, документація) в одному рішенні.\n"
                    )
                    mode = "explain_single_vs_team"

                # --- Special case 4: Multi-agent pipeline ---
                else:
                    pipeline_syns = ["pipeline", "пайплайн", "pipeline works", "як працює"]
                    multiagent_syns = ["multi-agent", "multi agent", "мультиагент"]
                    if (
                        (any(m in t for m in multiagent_syns))
                        and (any(p in t for p in pipeline_syns))
                    ):
                        output = (
                            "## Як працює наш multi-agent pipeline\n\n"
                            "1. Користувач формулює задачу.\n"
                            "2. Planner переформульовує запит і створює чіткий план дій.\n"
                            "3. Router або Supervisor обирають відповідних агентів (одного або команду).\n"
                            "4. Обрані агенти виконують завдання: аналізують, генерують текст, код або інші результати.\n"
                            "5. Critic оцінює чернетку, додає нотатки та теги якості.\n"
                            "6. Synthesizer об’єднує результати та формує фінальну відповідь.\n"
                            "7. Пам’ять і логи зберігають інформацію для майбутніх покращень.\n\n"
                            "Такий пайплайн дозволяє розділяти ролі, покращувати якість відповідей та ефективно вирішувати складні задачі."
                        )
                        mode = "explain_pipeline"

                    # --- Generic explain cases ---
                    else:
                        if force_structure:
                            output = (
                                f"## Питання\n{task}\n\n"
                                "## Як краще пояснити\n"
                                "- Коротко дайте визначення основних термінів.\n"
                                "- Опишіть мету або практичне призначення.\n"
                                "- Поясніть, чим це відрізняється від схожих підходів/понять.\n"
                                "- Наведіть простий приклад із життя або коду.\n"
                                "- Зробіть 1–2 речення-висновок.\n"
                            )
                            if expand_when_short:
                                output += "- Додайте практичну пораду або попередження про типову помилку.\n"
                            mode = "explain_scaffold_uk"
                        else:
                            output = (
                                "Щоб дати зрозуміле пояснення, корисно почати з короткого визначення, далі описати мету або для чого це використовується, навести один-два приклади і завершити коротким висновком. "
                                "Для цього запиту можна оформити відповідь саме в такому порядку."
                            )
                            if expand_when_short:
                                output += " Також варто додати практичну пораду або попередження про типову помилку, яку часто допускають у цій темі."
                            mode = "explain_generic_uk"

        # Застосовуємо стиль форматування з конфігів
        styled_output = _apply_style_to_text(output, config)
        return AgentResult(agent=self.name, output=styled_output, meta={"mode": mode})
