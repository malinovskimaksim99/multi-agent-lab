from typing import List, Optional, Tuple, Set, Dict

from .registry import list_agents, create_agent
from .base import Context, Memory
from db import get_solver_stats_by_task_type


DEFAULT_EXCLUDE: Set[str] = {"planner", "critic"}


def classify_task_type(task: str) -> str:
    """
    Дуже проста класифікація типу задачі за ключовими словами.
    Використовується тут, щоб враховувати історичну статистику з БД
    при ранжуванні агентів.
    """
    t = (task or "").lower()

    # Документація / README / інсталяція
    docs_markers = [
        "readme",
        "documentation",
        "docs",
        "guide",
        "installation",
        "інсталяц",
        "встанов",
        "документац",
        "гайд",
    ]
    if any(k in t for k in docs_markers):
        return "docs"

    # Пояснення / порівняння
    explain_markers = [
        "explain",
        "difference",
        "compare",
        "vs",
        "versus",
        "roles of",
        "summary",
        "summarize",
        "overview",
        "поясни",
        "що таке",
        "різниц",
        "порівняй",
        "огляд",
        "підсумуй",
    ]
    if any(k in t for k in explain_markers):
        return "explain"

    # Код / помилки / traceback
    code_markers = [
        "code",
        "bug",
        "error",
        "traceback",
        "exception",
        "syntaxerror",
        "stack trace",
        "код",
        "помилка",
        "скрипт",
        "стек",
    ]
    if any(k in t for k in code_markers):
        return "code"

    # Аналіз БД / запусків / датасету
    db_markers = [
        "бд",
        "database",
        "датасет",
        "dataset",
        "runs",
        "запусків",
        "аналіз запусків",
    ]
    if any(k in t for k in db_markers):
        return "db_analysis"

    # Планування
    plan_markers = [
        "plan",
        "planning",
        "roadmap",
        "outline",
        "план",
        "кроки",
        "стратег",
        "дорожня карта",
    ]
    if any(k in t for k in plan_markers):
        return "plan"

    # Мета-рівень / тренер
    meta_markers = [
        "trainer",
        "meta",
        "аналіз агентів",
        "аналіз запусків",
        "оптимізація агентів",
    ]
    if any(k in t for k in meta_markers):
        return "meta"

    return "other"


def rank_agents(
    task: str,
    memory: Memory,
    context: Optional[Context] = None,
    exclude: Optional[Set[str]] = None,
) -> List[Tuple[str, float]]:
    """
    Ранжує агентів за:
      1) їхнім власним can_handle (базовий скор),
      2) невеликим бонусом з урахуванням історичної статистики з БД
         для відповідного типу задачі (task_type).
    """
    ex = set(DEFAULT_EXCLUDE)
    if exclude:
        ex |= set(exclude)

    # Визначаємо тип задачі і тягнемо історичну статистику з БД
    task_type = classify_task_type(task)
    stats = get_solver_stats_by_task_type(task_type)
    max_count = max(stats.values()) if stats else 0

    scores: List[Tuple[str, float]] = []
    for name in list_agents():
        if name in ex:
            continue

        agent = create_agent(name)
        try:
            base_score = float(agent.can_handle(task, context))
        except Exception:
            base_score = 0.0

        # Бонус за історичну статистику по цьому типу задачі
        bonus = 0.0
        if stats and max_count > 0:
            cnt = stats.get(name, 0)
            if cnt > 0:
                frac = cnt / max_count  # від 0 до 1
                bonus = 0.2 * frac      # максимум +0.2 до скору

        score = base_score + bonus
        scores.append((name, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def pick_top(
    task: str,
    memory: Memory,
    k: int = 2,
    context: Optional[Context] = None,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    ranked = rank_agents(task, memory, context=context, exclude=exclude)
    return [name for name, score in ranked[:k] if score > 0]
