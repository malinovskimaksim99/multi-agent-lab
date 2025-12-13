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
    Повертає один з режимів: "agents", "roadmap", "dataset", "mixed".
    """
    t = task.lower()

    has_agent = any(k in t for k in ["агент", "agents", "agent"])
    has_roadmap = any(k in t for k in ["roadmap", "роадмап", "план розвитку"])
    has_dataset = any(k in t for k in ["датасет", "dataset", "приклад", "examples"])
    has_db = any(k in t for k in ["бд", "bd", "database", "runs"])

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
