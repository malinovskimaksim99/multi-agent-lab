# agents/head.py
from __future__ import annotations

from typing import Any, Dict

from .supervisor import Supervisor
from db import (
    list_projects,
    get_current_project,
    set_current_project,
    get_recent_errors,  # ми вже додавали це раніше
)


class HeadAgent:
    """
    Головний агент: приймає людські фрази і вирішує,
    що робити всередині multi-agent-lab.
    """

    def __init__(self) -> None:
        self.supervisor = Supervisor()

    def handle(self, text: str, memory: Any, *, auto: bool = True) -> str:
        """
        Основна точка входу.

        - text  — те, що ти написав у чаті
        - memory — наш об'єкт пам'яті (як у app.py / chat.py)
        - auto  — чи використовувати --auto-режим для Supervisor
        """
        clean = text.strip()
        lower = clean.lower()

        # --- 1. Команди про проєкти ---
        if lower in ("проєкт", "проект", "поточний проєкт"):
            proj = get_current_project()
            if not proj:
                return "Зараз немає активного проєкту."
            return (
                "Поточний проєкт:\n"
                f"- id={proj['id']}\n"
                f"- name={proj['name']}\n"
                f"- type={proj['type']}\n"
                f"- status={proj['status']}\n"
                f"- created_at={proj['created_at']}\n"
                f"- updated_at={proj['updated_at']}\n"
            )

        if lower in ("проєкти", "список проєктів", "проекты"):
            projects = list_projects()
            if not projects:
                return "У БД ще немає жодного проєкту."
            lines = ["Список проєктів:"]
            for p in projects:
                lines.append(
                    f"- id={p['id']} | name={p['name']} | type={p['type']} | status={p['status']}"
                )
            return "\n".join(lines)

        if lower.startswith("переключись на проєкт") or lower.startswith(
            "переключись на проект"
        ):
            # очікуємо щось типу: "переключись на проєкт Розробка"
            parts = clean.split(" ", 3)
            if len(parts) < 4:
                return "Скажи, на який проєкт переключитись, наприклад: 'переключись на проєкт Розробка'."
            project_name = parts[3].strip().strip('"“”')
            ok, msg = set_current_project(project_name)
            if not ok:
                return f"Не вдалося переключитись: {msg}"
            return f"Переключився на проєкт '{project_name}'."

        # --- 2. Аналіз помилок ---
        if lower in ("аналіз помилок", "аналіз помилки", "помилки з бд"):
            errors = get_recent_errors(limit=10)
            if not errors:
                return "У БД ще немає збережених помилок."
            lines = ["Останні помилки:"]
            for e in errors:
                lines.append(
                    f"[run_id={e['run_id']}] {e['error_type']}: {e['error_message']}"
                )
            return "\n".join(lines)

        # --- 3. Письменницькі команди (простий варіант-плайсхолдер) ---
        if lower in ("письменницькі проекти", "письменницькі проєкти"):
            # Поки просто показуємо всі проєкти типу 'writing'
            projects = list_projects(project_type="writing")
            if not projects:
                return "Письменницьких проєктів поки немає."
            lines = ["Письменницькі проєкти:"]
            for p in projects:
                lines.append(f"- id={p['id']} | name={p['name']} | status={p['status']}")
            return "\n".join(lines)

        # --- 4. За замовчуванням: це звичайна задача → Supervisor ---
        # Тут ми поводимось як app.py --task "...".
        context: Dict[str, Any] = {}
        result = self.supervisor.run(clean, memory, context, auto=auto)
        return result