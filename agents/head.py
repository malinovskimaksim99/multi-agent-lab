# agents/head.py
from __future__ import annotations

from typing import Any, Dict

from .supervisor import Supervisor

from db import (
    get_projects,
    get_current_project,
    set_current_project,
    get_recent_errors,  # ми вже додавали це раніше
)

from .head_profile import build_head_system_prompt
from ollama_client import call_head_llm


class HeadAgent:
    """
    Головний агент: приймає людські фрази і вирішує,
    що робити всередині multi-agent-lab.
    """

    def __init__(self) -> None:
        self.supervisor = Supervisor()
        # Паспорт / системний промпт для головного агента
        self.system_prompt = build_head_system_prompt()
        self.use_ollama_head = True  # цей прапор поки не використовується, але буде корисний далі

    def ask_llm(self, user_text: str) -> str:
        """
        Виклик Qwen через Ollama як "мозок" HeadAgent-а.

        Поки що:
        - використовуємо системний промпт з head_profile,
        - звертаємось до моделі qwen3-vl:8b,
        - не обмежуємо max_tokens (щоб не обрізати відповідь штучно),
        - якщо щось пішло не так — повертаємо зрозуміле повідомлення про помилку.
        """
        try:
            reply = call_head_llm(
                user_text=user_text,
                system_prompt=self.system_prompt,
                model="qwen3-vl:8b",
                temperature=0.2,
                # max_tokens не задаємо: довжину контролює сама модель,
                # а лаконічність ми задаємо в HEAD_SYSTEM_PROMPT.
            )
        except Exception as e:
            return f"[HeadAgent/Qwen помилка: {e}]"

        if not reply:
            return "[HeadAgent/Qwen не повернув текст відповіді]"

        reply = reply.strip()
        if not reply:
            return "[HeadAgent/Qwen повернув лише порожній рядок]"

        return reply

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

            # Якщо get_current_project() повертає вже готовий текст (рядок) — просто віддаємо його.
            if isinstance(proj, str):
                return proj

            # Інакше очікуємо dict з полями проєкту.
            try:
                return (
                    "Поточний проєкт:\n"
                    f"- id={proj['id']}\n"
                    f"- name={proj['name']}\n"
                    f"- type={proj['type']}\n"
                    f"- status={proj['status']}\n"
                    f"- created_at={proj['created_at']}\n"
                    f"- updated_at={proj['updated_at']}\n"
                )
            except Exception:
                # Фолбек на випадок, якщо структура інша — хоч щось осмислене показати.
                return f"Поточний проєкт: {proj}"

        if lower in ("проєкти", "список проєктів", "проекты"):
            projects = get_projects()
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
            all_projects = get_projects()
            projects = [p for p in all_projects if p.get("type") == "writing"]
            if not projects:
                return "Письменницьких проєктів поки немає."
            lines = ["Письменницькі проєкти:"]
            for p in projects:
                lines.append(f"- id={p['id']} | name={p['name']} | status={p['status']}")
            return "\n".join(lines)

        # --- 4. За замовчуванням: це звичайна "людська" задача → Qwen як голова ---
        #
        # На цьому етапі:
        # - спеціальні службові команди ("проєкт", "проєкти", "аналіз помилок", "письменницькі проєкти")
        #   обробляються вище чистим Python-кодом;
        # - усі інші запити йдуть у ask_llm(), тобто до Qwen як мозку HeadAgent-а.
        #
        # Поки що ми не просимо Qwen самостійно викликати Supervisor чи інші агенти —
        # це буде окремий етап (інструменти / BossAgent).
        reply = self.ask_llm(clean)
        return reply