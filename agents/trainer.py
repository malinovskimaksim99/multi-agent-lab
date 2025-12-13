from typing import Optional, Any, Dict, List
from collections import Counter, defaultdict
import re
import json

from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent
from .router import classify_task_type
from db import get_recent_runs


def _extract_limit_from_text(text: str, default: int = 50) -> int:
    """
    Витягує число із тексту (типу 'проаналізуй 100 запусків').

    Якщо числа немає – повертає default.
    Обмежує значення розумним діапазоном [1, 500].
    """
    m = re.search(r"(\d+)", text)
    if not m:
        return default
    try:
        n = int(m.group(1))
    except ValueError:
        return default
    return max(1, min(n, 500))


@register_agent
class TrainerAgent(BaseAgent):
    """
    Trainer / Meta-агент.

    Читає останні запуски з БД (runs.db) і формує звіт:
      - по агентах (скільки запусків, скільки з тегами),
      - по тегах критика,
      - базові рекомендації, де варто посилити структуру / кроки / приклади.
    """

    name = "trainer"
    description = "Аналізує запускі з БД (runs.db) і дає рекомендації для покращення агентів."

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        t = task.lower()

        markers = [
            "аналіз бд",
            "аналіз запусків",
            "аналіз запускiв",
            "проаналізуй запуск",
            "проаналізуй бд",
            "trainer",
            "meta agent",
            "meta-агент",
            "оптимізуй агент",
            "optimize agents",
            "analyze db runs",
            "db runs analysis",
            "runs analysis",
        ]

        if any(m in t for m in markers):
            return 0.95

        # Загальна евристика: якщо є "аналіз" + згадки про БД або запусків,
        # вважаємо, що це запит до тренера.
        if "аналіз" in t and ("бд" in t or "database" in t or "запуск" in t or "запусків" in t or "runs" in t):
            return 0.9

        # якщо є згадки про eval / теги / критику – теж можемо підхопити
        soft_markers = [
            "critique tags",
            "теги крита",
            "eval результати",
            "результати eval",
            "помилки агентів",
        ]
        if any(m in t for m in soft_markers):
            return 0.7

        return 0.1

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        t = task.lower().strip()

        limit = _extract_limit_from_text(t, default=50)

        try:
            runs = get_recent_runs(limit=limit)
        except Exception as e:
            output = (
                "Не вдалося прочитати запускі з БД (runs.db).\n\n"
                f"Помилка: {e}\n\n"
                "Перевір, що файл `runs.db` існує в корені проєкту і що структура таблиці відповідає очікуваній."
            )
            return AgentResult(
                agent=self.name,
                output=output,
                meta={"mode": "trainer_error"},
            )

        if not runs:
            output = (
                "У БД ще немає збережених запусків, тому аналіз поки що порожній.\n"
                "Запусти кілька задач через `app.py` або `chat.py`, а потім повтори запит до тренера."
            )
            return AgentResult(
                agent=self.name,
                output=output,
                meta={"mode": "trainer_empty"},
            )

        total = len(runs)

        # Статистика по типах задач (task_type)
        type_total: Counter = Counter()
        type_by_solver: Dict[str, Counter] = defaultdict(Counter)

        # Статистика по агентах
        solver_total: Counter = Counter()
        solver_with_tags: Counter = Counter()

        # Статистика по тегах
        tag_total: Counter = Counter()
        tag_examples: Dict[str, List[str]] = defaultdict(list)

        for r in runs:
            solver = r.get("solver_agent") or "unknown"
            solver_total[solver] += 1

            # Текст задачі для аналізу/прикладів
            task_text = (r.get("task") or "").strip()

            # Тип задачі: спочатку пробуємо взяти з БД, а якщо його немає –
            # визначаємо через classify_task_type за текстом задачі.
            task_type = r.get("task_type") or classify_task_type(task_text)
            if not task_type:
                task_type = "other"

            type_total[task_type] += 1
            type_by_solver[task_type][solver] += 1

            tags = r.get("critique_tags") or []
            if tags:
                solver_with_tags[solver] += 1
                tag_total.update(tags)

                # кілька прикладів задач для кожного тегу
                short_text = task_text
                if len(short_text) > 80:
                    short_text = short_text[:77] + "..."
                for tag in tags:
                    if len(tag_examples[tag]) < 3:
                        tag_examples[tag].append(short_text)

        lines: List[str] = []
        lines.append(f"## Аналіз останніх {total} запусків з БД\n")

        # Підсумок по агентах
        lines.append("### Підсумок по агентах")
        for solver, cnt in solver_total.most_common():
            bad = solver_with_tags.get(solver, 0)
            if cnt == 0:
                ratio = 0.0
            else:
                ratio = bad / cnt
            lines.append(f"- **{solver}**: {cnt} запусків, з тегами: {bad} (≈ {ratio:.0%})")
        lines.append("")

        # Підсумок по типах задач
        lines.append("### Підсумок по типах задач (task_type)")
        if not type_total:
            lines.append("Поки що не виявлено типів задач (task_type) у запусків.")
        else:
            for ttype, cnt in type_total.most_common():
                lines.append(f"- **{ttype}**: {cnt} запусків")
        lines.append("")

        # Агенти по типах задач
        lines.append("### Агенти по типах задач")
        if not type_by_solver:
            lines.append("Немає достатньо інформації про відповідність агентів до типів задач.")
        else:
            for ttype, solver_cnt in type_by_solver.items():
                parts = [f"{s}: {c}" for s, c in solver_cnt.most_common()]
                joined = ", ".join(parts)
                lines.append(f"- **{ttype}**: {joined}")
        lines.append("")

        # Проблемні теги
        lines.append("### Проблемні теги критика")
        if not tag_total:
            lines.append(
                "У останніх запусків немає тегів критика — явних проблем не зафіксовано. "
                "Це означає, що відповіді виглядають достатньо структурованими і повними."
            )
        else:
            for tag, cnt in tag_total.most_common():
                examples = tag_examples.get(tag) or []
                if examples:
                    ex_str = "; приклади задач: " + " | ".join(examples)
                else:
                    ex_str = ""
                lines.append(f"- **{tag}**: {cnt} раз(ів){ex_str}")
        lines.append("")

        # Рекомендації
        lines.append("### Попередні рекомендації")

        if "missing_structure" in tag_total:
            lines.append(
                "- Посилити структуру відповідей (заголовки, підзаголовки, списки) "
                "для тих типів задач, де часто зʼявляється тег `missing_structure`."
            )
        if "missing_steps" in tag_total:
            lines.append(
                "- Для інструкцій (особливо встановлення/кроки) завжди додавати нумерований список "
                "та, за можливості, чекліст або явний розділ `Кроки`."
            )
        if "too_short" in tag_total:
            lines.append(
                "- Для складніших explain/overview задач давати трохи більше деталей та прикладів, "
                "щоб уникнути тегу `too_short`."
            )
        if "meta_template" in tag_total:
            lines.append(
                "- Переконатися, що агенти повертають готову відповідь для користувача, "
                "а не внутрішні meta-інструкції (template-підказки)."
            )

        if not any(tag in tag_total for tag in ["missing_structure", "missing_steps", "too_short", "meta_template"]):
            lines.append(
                "- На основі останніх запусків критичних проблем не видно. "
                "Можна поступово додавати нові типи задач та спостерігати за тегами критика."
            )

        # Чернетка пропозицій для конфігів агентів (agent_configs)
        suggestions: Dict[str, Dict[str, Any]] = {}

        # 1) На основі типів задач: для кожного task_type — агент, який найчастіше його брав
        for ttype, solver_cnt in type_by_solver.items():
            if not solver_cnt:
                continue
            top_solver, _ = solver_cnt.most_common(1)[0]
            agent_cfg = suggestions.setdefault(top_solver, {})
            preferred = agent_cfg.setdefault("preferred_task_types", [])
            if ttype not in preferred:
                preferred.append(ttype)

        # 2) На основі проблемних тегів критика — базові рекомендації
        if "missing_structure" in tag_total:
            writer_cfg = suggestions.setdefault("writer", {})
            writer_cfg["force_structure"] = True

        if "missing_steps" in tag_total:
            writer_cfg = suggestions.setdefault("writer", {})
            writer_cfg["emphasize_steps"] = True

        if "too_short" in tag_total:
            explainer_cfg = suggestions.setdefault("explainer", {})
            explainer_cfg["min_detail_level"] = "medium"

        if "meta_template" in tag_total:
            analyst_cfg = suggestions.setdefault("analyst", {})
            analyst_cfg["avoid_meta_templates"] = True

        # 3) Додаємо блок з JSON-пропозиціями в текст звіту
        lines.append("### Пропозиції для агентів (чернетка)")
        if not suggestions:
            lines.append("Поки що немає явних пропозицій для зміни конфігів агентів.")
        else:
            lines.append("Нижче — чернетка можливих змін для `agent_configs` у форматі JSON:")
            try:
                suggestions_json = json.dumps(suggestions, ensure_ascii=False, indent=2)
            except TypeError:
                suggestions_json = "{}"
            lines.append("")
            lines.append("```json")
            lines.append(suggestions_json)
            lines.append("```")
        lines.append("")

        output = "\n".join(lines)

        meta: Dict[str, Any] = {
            "mode": "trainer_report",
            "total_runs": total,
            "top_solvers": solver_total.most_common(5),
            "top_tags": tag_total.most_common(5),
            "config_suggestions": suggestions,
        }

        return AgentResult(
            agent=self.name,
            output=output,
            meta=meta,
        )