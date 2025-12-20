# agents/head.py
from __future__ import annotations

from typing import Any, Dict, Optional

from .supervisor import Supervisor

from db import (
    get_projects,
    get_current_project,
    set_current_project,
    get_recent_errors,  # ми вже додавали це раніше
    add_head_note,
    get_head_notes,
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
        - звертаємось до моделі qwen3:8b,
        - не обмежуємо max_tokens (щоб не обрізати відповідь штучно),
        - якщо щось пішло не так — повертаємо зрозуміле повідомлення про помилку.
        """
        try:
            reply = call_head_llm(
                user_text=user_text,
                system_prompt=self.system_prompt,
                model="qwen3:8b",
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

    def _get_current_project_name(self) -> Optional[str]:
        """Повертає назву поточного проєкту (якщо є), інакше None."""
        try:
            proj = get_current_project()
        except Exception:
            return None

        if not proj:
            return None

        # Іноді це може бути вже готовий текст/рядок.
        if isinstance(proj, str):
            # Якщо там просто "Розробка" — ок. Якщо це довгий опис — все одно краще
            # нічого не логувати, ніж логувати в дивну назву.
            s = proj.strip()
            if 1 <= len(s) <= 80 and "\n" not in s:
                return s
            return None

        if isinstance(proj, dict):
            name = proj.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()

        return None

    def _log_head_note(self, user_text: str, reply: str) -> None:
        """Записує короткий лог взаємодії HeadAgent-а в head_notes.

        Важливо: не ламаємо чат, якщо БД/схема/параметри не співпали — просто мовчки ігноруємо.
        """
        project_name = self._get_current_project_name()
        if not project_name:
            return

        note_text = f"USER: {user_text.strip()}\n\nHEAD: {reply.strip()}"

        # Прагнемо бути сумісними з різними сигнатурами add_head_note(...)
        try:
            add_head_note(
                project_name,
                scope="project",
                note_type="interaction",
                note=note_text,
                importance=1,
                tags="head,interaction",
                source="head_agent",
                author="head",
            )
            return
        except TypeError:
            # Можливо, функція приймає інші імена параметрів/менше аргументів.
            pass
        except Exception:
            return

        try:
            # Мінімальний фолбек: тільки те, що майже напевно є.
            add_head_note(project_name, note_text)
        except Exception:
            return

    def _format_head_notes(self, notes: Any) -> str:
        """Гарне форматування списку нотаток для відповіді в чаті."""
        if not notes:
            return "Нотаток HeadAgent-а поки немає."

        # Очікуємо list[dict], але робимо фолбеки
        if not isinstance(notes, list):
            return f"Нотатки: {notes}"

        lines = ["Останні head нотатки:"]
        for i, n in enumerate(notes, start=1):
            if isinstance(n, dict):
                created = n.get("created_at") or n.get("updated_at") or ""
                note_type = n.get("note_type") or ""
                note = (n.get("note") or "").strip()
                if len(note) > 400:
                    note = note[:400].rstrip() + "…"
                meta = " | ".join([p for p in [created, note_type] if p])
                if meta:
                    lines.append(f"{i}) [{meta}]\n{note}")
                else:
                    lines.append(f"{i}) {note}")
            else:
                lines.append(f"{i}) {n}")
        return "\n\n".join(lines)

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

        # --- 4. Head нотатки ---
        if lower in (
            "head нотатки",
            "покажи head нотатки",
            "покажи нотатки head",
            "show head notes",
        ):
            project_name = self._get_current_project_name()
            if not project_name:
                return "Немає активного проєкту, тому і head нотатки показати не можу."

            try:
                notes = get_head_notes(project_name=project_name, limit=5)
            except TypeError:
                # Фолбек для іншої сигнатури
                try:
                    notes = get_head_notes(project_name, 5)
                except Exception as e:
                    return f"Не вдалося прочитати head нотатки: {e}"
            except Exception as e:
                return f"Не вдалося прочитати head нотатки: {e}"

            return self._format_head_notes(notes)

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
        self._log_head_note(clean, reply)
        return reply