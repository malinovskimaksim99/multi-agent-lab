

from typing import Optional, Any, Dict, List
from collections import Counter, defaultdict
import re

from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent
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

        # Статистика по агентах
        solver_total: Counter = Counter()
        solver_with_tags: Counter = Counter()

        # Статистика по тегах
        tag_total: Counter = Counter()
        tag_examples: Dict[str, List[str]] = defaultdict(list)

        for r in runs:
            solver = r.get("solver_agent") or "unknown"
            solver_total[solver] += 1

            tags = r.get("critique_tags") or []
            if tags:
                solver_with_tags[solver] += 1
                tag_total.update(tags)

                # кілька прикладів задач для кожного тегу
                task_text = r.get("task", "") or ""
                if len(task_text) > 80:
                    task_text = task_text[:77] + "..."
                for tag in tags:
                    if len(tag_examples[tag]) < 3:
                        tag_examples[tag].append(task_text)

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

        output = "\n".join(lines)

        meta: Dict[str, Any] = {
            "mode": "trainer_report",
            "total_runs": total,
            "top_solvers": solver_total.most_common(5),
            "top_tags": tag_total.most_common(5),
        }

        return AgentResult(
            agent=self.name,
            output=output,
            meta=meta,
        )