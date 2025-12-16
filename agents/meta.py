from typing import Optional, Dict, Any, List
from pathlib import Path
import sqlite3
import json

from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent
from db import (
    get_recent_runs,
    get_dataset_examples,
    get_solver_stats_by_task_type,
)


def _classify_meta_mode(task: str) -> str:
    """
    Дуже проста евристика: що саме користувач хоче від MetaAgent.
    Повертає один з режимів: "agents", "roadmap", "dataset", "mixed", "training".
    """
    t = task.lower()

    has_training = any(
        k in t
        for k in [
            "meta тренув",
            "мета тренув",
            "meta training",
            "training tasks",
            "тренувальн задач",
            "тренувальні задачі",
        ]
    )

    has_agent = any(k in t for k in ["агент", "agents", "agent"])
    has_roadmap = any(k in t for k in ["roadmap", "роадмап", "план розвитку"])
    has_dataset = any(k in t for k in ["датасет", "dataset", "приклад", "examples"])
    has_db = any(k in t for k in ["бд", "bd", "database", "runs"])

    if has_training:
        return "training"
    if has_roadmap:
        return "roadmap"
    if has_agent:
        return "agents"
    if has_dataset:
        return "dataset"
    if has_db:
        return "mixed"
    return "agents"


def _format_agent_stats(runs: List[Dict[str, Any]]) -> str:
    """
    Робить короткий звіт по агентах і типах задач на основі runs.
    """
    by_agent: Dict[str, Dict[str, Any]] = {}
    by_type: Dict[str, int] = {}

    for r in runs:
        solver = r.get("solver_agent") or "unknown"
        tags = r.get("tags") or []
        task_type = r.get("task_type") or "other"

        s = by_agent.setdefault(solver, {"runs": 0, "tagged": 0})
        s["runs"] += 1
        if tags:
            s["tagged"] += 1

        by_type[task_type] = by_type.get(task_type, 0) + 1

    lines: List[str] = []
    lines.append("### Підсумок по агентах")
    if not by_agent:
        lines.append("- Немає достатньо даних про запуски.")
    else:
        for name, s in sorted(by_agent.items(), key=lambda kv: kv[1]["runs"], reverse=True):
            runs_cnt = s["runs"]
            tagged = s["tagged"]
            perc = (tagged / runs_cnt * 100.0) if runs_cnt > 0 else 0.0
            lines.append(f"- **{name}**: {runs_cnt} запусків, з тегами: {tagged} (≈ {perc:.0f}%)")

    lines.append("")
    lines.append("### Типи задач (task_type)")
    if not by_type:
        lines.append("- Немає даних про типи задач.")
    else:
        for ttype, cnt in sorted(by_type.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"- **{ttype}**: {cnt} запусків")

    return "\n".join(lines)


def _load_all_agent_configs_from_db() -> Dict[str, Dict[str, Any]]:
    """
    Зчитує всі конфіги агентів з таблиці agent_configs у runs.db.
    Формат виходу: {agent_name: {config_key: parsed_value, ...}, ...}
    """
    db_path = Path("runs.db")
    if not db_path.exists():
        return {}

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            "SELECT agent_name, config_key, config_value FROM agent_configs"
        )
        rows = cur.fetchall()
    except Exception:
        return {}
    finally:
        try:
            conn.close()
        except Exception:
            pass

    cfgs: Dict[str, Dict[str, Any]] = {}
    for agent_name, key, value in rows:
        parsed: Any = value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                parsed = value
        cfgs.setdefault(agent_name, {})[key] = parsed

    return cfgs


def _format_configs_summary() -> str:
    """
    Короткий огляд конфігів агентів з agent_configs.
    """
    cfgs = _load_all_agent_configs_from_db()
    lines: List[str] = ["### Конфіги агентів"]
    if not cfgs:
        lines.append("- Поки що немає явних конфігів у БД.")
        return "\n".join(lines)

    for agent_name, cfg in sorted(cfgs.items()):
        lines.append(f"- **{agent_name}**: {cfg}")

    return "\n".join(lines)


def _format_dataset_summary(limit: int = 50) -> str:
    """
    Огляд датасету: скільки прикладів і які мітки частіше зустрічаються.
    """
    examples = get_dataset_examples(limit=limit)
    lines: List[str] = ["### Датасет прикладів"]

    if not examples:
        lines.append("- Поки що немає прикладів у датасеті.")
        return "\n".join(lines)

    by_label: Dict[str, int] = {}
    for ex in examples:
        label = (ex.get("label") or "unknown").lower()
        by_label[label] = by_label.get(label, 0) + 1

    total = len(examples)
    for label, cnt in sorted(by_label.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- **{label}**: {cnt} прикладів")

    lines.append(f"- Всього прикладів у вибірці: {total}")
    return "\n".join(lines)


def _format_roadmap() -> str:
    """
    Узагальнений roadmap розвитку multi-agent-lab на основі поточної архітектури.
    """
    lines: List[str] = [
        "## Roadmap розвитку multi-agent-lab",
        "",
        "### 1. Дані та спостереження",
        "- Продовжувати запускати реальні задачі й накопичувати історію в `runs`.",
        "- Позначати хороші / проблемні / цікаві відповіді у `dataset_examples`.",
        "- Системно використовувати теги критика (too_short, missing_structure тощо).",
        "",
        "### 2. Налаштування агентів через конфіги",
        "- Уточнити `preferred_task_types` для ключових агентів (analyst, coder, trainer, writer, explainer).",
        "- Додати мʼякі параметри стилю: `style`, `max_lines`, `force_structure`.",
        "- Тримати конфіги у `agent_configs` як єдине джерело правди.",
        "",
        "### 3. Meta-аналіз і автооновлення",
        "- Запускати Trainer/MetaAgent для аналізу останніх N запусків.",
        "- На основі статистики та датасету пропонувати зміни у `agent_configs`.",
        "- Переглядати пропозиції в чаті й одним кроком застосовувати їх.",
        "",
        "### 4. Розширення екосистеми агентів",
        "- Додавати спеціалізованих агентів під окремі домени (код, документація, аналітика, майбутні SDR тощо).",
        "- Використовувати MetaAgent, щоб підлаштовувати їхні ролі та стилі.",
        "",
        "### 5. Інтеграції та API (коли буде актуально)",
        "- Додати можливість окремим агентам ходити в зовнішні API для збору даних.",
        "- Обмежувати це через конфіги й окремі прапорці безпеки.",
    ]
    return "\n".join(lines)


def _format_training_recommendations(
    runs: List[Dict[str, Any]],
    cfgs: Dict[str, Dict[str, Any]],
    limit: int = 5,
) -> str:
    """
    Формує список конкретних тренувальних кроків на основі історії запусків
    та наявних конфігів агентів.
    """
    lines: List[str] = ["### Рекомендовані тренувальні кроки"]

    if not runs:
        lines.append("- Поки що немає історії запусків — спочатку запусти кілька задач.")
        return "\n".join(lines)

    by_agent: Dict[str, Dict[str, Any]] = {}
    by_type: Dict[str, int] = {}
    tagged_total = 0

    for r in runs:
        agent = (r.get("solver_agent") or "unknown").lower()
        ttype = (r.get("task_type") or "other").lower()
        tags = r.get("tags") or []

        s = by_agent.setdefault(agent, {"runs": 0, "tagged": 0})
        s["runs"] += 1
        if tags:
            s["tagged"] += 1
            tagged_total += 1

        by_type[ttype] = by_type.get(ttype, 0) + 1

    recs: List[str] = []

    # 1) Якщо датасет майже порожній — пропонуємо його наповнювати
    examples = get_dataset_examples(limit=200)
    if len(examples) < 10:
        recs.append(
            "Додай ще кілька прикладів у датасет (good/bad/edge cases), "
            "особливо для типових задач: пояснення, плани, аналіз БД."
        )

    # 2) Якщо немає тегів критика — заохочуємо їх використовувати
    if tagged_total == 0:
        recs.append(
            "Почни активніше позначати відповіді тегами через критика "
            "(too_short, missing_structure тощо), щоб система бачила, де є проблеми."
        )

    # 3) Пошук агентів з найменшою кількістю запусків
    if by_agent:
        sorted_agents = sorted(by_agent.items(), key=lambda kv: kv[1]["runs"])
        low_name, low_stats = sorted_agents[0]
        high_name, high_stats = sorted_agents[-1]
        if low_stats["runs"] < high_stats["runs"]:
            recs.append(
                f"Додай кілька задач спеціально для агента `{low_name}`, "
                "щоб краще зрозуміти його поведінку."
            )

    # 4) Перевірка ключових агентів на наявність конфігів
    important_agents = ("analyst", "coder", "trainer")
    for important in important_agents:
        if important not in cfgs:
            recs.append(
                f"Додай базовий config для агента `{important}` у agent_configs "
                "(наприклад, preferred_task_types, force_structure)."
            )
            break

    if not recs:
        recs.append(
            "Продовжуй запускати різні задачі — поки що система виглядає збалансованою."
        )

    for r in recs[:limit]:
        lines.append(f"- {r}")

    return "\n".join(lines)


def _format_training_tasks(
    runs: List[Dict[str, Any]],
    cfgs: Dict[str, Dict[str, Any]],
    max_types: int = 3,
    per_type: int = 3,
) -> str:
    """
    Генерує список конкретних тренувальних задач на основі:
      - історії запусків (які task_type зустрічаються рідше / частіше);
      - конфігів агентів (preferred_task_types),
      - вже існуючих шаблонів задач.

    Ці задачі поки що тільки повертаються текстом
    (наступним кроком можна буде додавати їх у датасет автоматично).
    """
    lines: List[str] = ["### Пропоновані тренувальні задачі"]

    # Якщо немає історії — даємо базовий стартовий набір задач
    if not runs:
        lines.append("- Поки що немає історії запусків — пропоную базові задачі для старту:")
        lines.append("")
        lines.append("#### Для документації (docs)")
        lines.append("1. Write a short README outline for the multi-agent-lab project.")
        lines.append("2. Напиши розділ README про встановлення та запуск multi-agent-lab.")
        lines.append("")
        lines.append("#### Для пояснень (explain)")
        lines.append("1. Поясни різницю між Planner, Analyst і Critic у multi-agent-lab.")
        lines.append("2. Поясни, як розібратися з traceback помилки в Python-скрипті.")
        lines.append("")
        lines.append("#### Для планування (plan)")
        lines.append("1. Склади короткий план розвитку multi-agent-lab на 3–5 кроків.")
        return "\n".join(lines)

    # Статистика по типах задач
    by_type: Dict[str, int] = {}
    for r in runs:
        ttype = (r.get("task_type") or "other").lower()
        by_type[ttype] = by_type.get(ttype, 0) + 1

    # Цікавлять нас перш за все такі типи
    interesting_types = ["docs", "explain", "code", "plan", "db_analysis"]

    # Сортуємо за кількістю запусків (рідкісніші типи вище)
    sorted_types = sorted(
        [t for t in interesting_types if t in by_type],
        key=lambda t: by_type.get(t, 0),
    )

    # Якщо якихось взагалі не було — додаємо їх наприкінці списку з нульовою кількістю
    for t in interesting_types:
        if t not in by_type and t not in sorted_types:
            sorted_types.append(t)

    # Обмежуємося max_types типами
    selected_types = sorted_types[:max_types] or interesting_types[:max_types]

    def _find_main_agent_for_type(ttype: str) -> Optional[str]:
        # Шукаємо агента, у якого preferred_task_types містить цей тип
        for agent_name, cfg in cfgs.items():
            prefs = cfg.get("preferred_task_types") or []
            if isinstance(prefs, list) and ttype in [str(p).lower() for p in prefs]:
                return agent_name
        # Фолбек вручну
        fallback: Dict[str, str] = {
            "docs": "docs",
            "explain": "explainer",
            "code": "coder",
            "plan": "analyst",
            "db_analysis": "trainer",
        }
        return fallback.get(ttype)

    for ttype in selected_types:
        main_agent = _find_main_agent_for_type(ttype)
        title_suffix = f" (головний агент: {main_agent})" if main_agent else ""
        if ttype == "docs":
            lines.append(f"#### Для документації (docs){title_suffix}")
            lines.append("1. Write a short README outline for the multi-agent-lab project.")
            lines.append("2. Напиши розділ README про встановлення та запуск multi-agent-lab.")
            lines.append("3. Напиши розділ README про використання multi-agent-lab з прикладами команд.")
        elif ttype == "explain":
            lines.append(f"#### Для пояснень (explain){title_suffix}")
            lines.append("1. Поясни різницю між single-solver та team-solver режимами.")
            lines.append("2. Поясни ролі Planner, Analyst, Critic у multi-agent-lab.")
            lines.append("3. Поясни, як розібратися з traceback помилки в Python-скрипті.")
        elif ttype == "code":
            lines.append(f"#### Для коду (code){title_suffix}")
            lines.append("1. Поясни, як знайти і виправити типову помилку TypeError у Python.")
            lines.append("2. Опиши кроки налагодження помилки ImportError в невеликому скрипті.")
            lines.append("3. Запропонуй покроковий підхід до рефакторингу довгої функції в Python.")
        elif ttype == "plan":
            lines.append(f"#### Для планування (plan){title_suffix}")
            lines.append("1. Склади короткий план розвитку multi-agent-lab на 3–5 кроків.")
            lines.append("2. Запропонуй план додавання нового спеціалізованого агента в систему.")
            lines.append("3. Склади план покращення якості відповідей coder-агента.")
        elif ttype == "db_analysis":
            lines.append(f"#### Для аналізу БД (db_analysis){title_suffix}")
            lines.append("1. Зроби аналіз 50 останніх запусків у БД і опиши, які агенти використовуються найчастіше.")
            lines.append("2. Зроби аналіз запусків і запропонуй, для яких агентів варто додати або змінити конфіги.")
            lines.append("3. Запропонуй критерії, за якими варто позначати запуски як good / bad / edge-case у датасеті.")
        lines.append("")

    return "\n".join(lines)


@register_agent
class MetaAgent(BaseAgent):
    """
    MetaAgent: високорівневий аналітик системи агентів.

    Він не вирішує прикладні задачі напряму, а:
      - аналізує історію запусків (`runs`);
      - дивиться на датасет (`dataset_examples`);
      - читає конфіги агентів (`agent_configs`);
      - формує рекомендації й roadmap для розвитку системи.
    """

    name = "meta"
    description = (
        "Meta-агент, який аналізує роботу всієї системи: історію запусків, датасет і конфіги агентів, "
        "та пропонує покращення."
    )

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        t = (task or "").lower()

        # Спочатку окрема перевірка для meta-запитів про тренування / тренувальні задачі
        training_markers = ["meta тренув", "мета тренув", "meta training", "training tasks"]
        if any(k in t for k in training_markers):
            return 0.99

        markers = [
            "meta agent",
            "meta-agent",
            "metaagent",
            "мета-агент",
            "мета агент",
            "аналіз агентів",
            "статистика агентів",
            "оптимізація агентів",
            "оптимизация агентов",
            "agent analysis",
            "agent stats",
            "meta тренув",
            "мета тренув",
            "meta training",
            "training tasks",
            "тренувальн задач",
            "тренувальні задачі",
        ]
        if any(m in t for m in markers):
            return 0.98

        if any(k in t for k in ["roadmap", "роадмап", "план розвитку"]):
            return 0.9

        if "аналіз" in t and "бд" in t:
            return 0.85

        if "датасет" in t or "dataset" in t:
            return 0.8

        # За замовчуванням — низький пріоритет (MetaAgent не перехоплює звичайні задачі)
        return 0.05

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        mode = _classify_meta_mode(task)
        runs = get_recent_runs(limit=100)
        cfgs = _load_all_agent_configs_from_db()

        sections: List[str] = []
        sections.append(f"## Meta-звіт по системі агентів\n\n### Вхідна задача\n{task}\n")

        if mode in ("agents", "mixed"):
            sections.append(_format_agent_stats(runs))

        if mode in ("dataset", "mixed"):
            sections.append("")
            sections.append(_format_dataset_summary(limit=50))

        # Конфіги корисні в будь-якому режимі
        sections.append("")
        sections.append(_format_configs_summary())

        # Якщо задача схожа на запит тренувальних задач — додаємо окремий блок із задачами
        if mode == "training":
            sections.append("")
            sections.append(_format_training_tasks(runs, cfgs))

        # Конкретні тренувальні кроки на основі історії запусків і конфігів
        sections.append("")
        sections.append(_format_training_recommendations(runs, cfgs))

        if mode == "roadmap":
            sections.append("")
            sections.append(_format_roadmap())

        output = "\n\n".join(s for s in sections if s)

        return AgentResult(
            agent=self.name,
            output=output,
            meta={
                "mode": "meta_analysis",
                "meta_mode": mode,
            },
        )
