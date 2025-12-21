# agents/head.py
from __future__ import annotations

import os
import inspect
import json
import subprocess

from typing import Any, Dict, Optional

from .supervisor import Supervisor

from db import (
    get_projects,
    get_current_project,
    set_current_project,
    get_recent_errors,  # ми вже додавали це раніше
    add_head_note,
    get_head_notes,
    get_llm_config,
)

from .head_profile import build_head_system_prompt
from llm_client import chat_openai_compat, env_default_base_url


class HeadAgent:
    """
    Головний агент: приймає людські фрази і вирішує,
    що робити всередині multi-agent-lab.
    """

    def __init__(self) -> None:
        self.supervisor = Supervisor()
        # Паспорт / системний промпт для головного агента
        self.system_prompt = build_head_system_prompt()

    def ask_llm(self, user_text: str) -> str:
        """
        Виклик LM Studio / OpenAI-compatible як "мозок" HeadAgent-а.

        Поки що:
        - використовуємо системний промпт з head_profile,
        - звертаємось до моделі з HEAD_MODEL (за замовчуванням qwen2.5-7b-instruct-1m),
        - не обмежуємо max_tokens (щоб не обрізати відповідь штучно),
        - якщо щось пішло не так — повертаємо зрозуміле повідомлення про помилку.
        """
        try:
            cfg = self._get_llm_config_for_current_project()
            base_url = os.getenv(
                "LMSTUDIO_BASE_URL",
                cfg.get("base_url", "http://127.0.0.1:1234/v1"),
            )
            model = os.getenv(
                "HEAD_MODEL",
                cfg.get("head_model", "qwen2.5-7b-instruct-1m"),
            )
            reply = chat_openai_compat(
                base_url=base_url,
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.2,
                timeout_s=int(os.getenv("LLM_TIMEOUT_S", "120")),
            )
        except Exception as e:
            return f"[HeadAgent/LM Studio помилка: {e}]"

        if not reply:
            return "[HeadAgent/LM Studio не повернув текст відповіді]"

        reply = reply.strip()
        if not reply:
            return "[HeadAgent/LM Studio повернув лише порожній рядок]"

        return reply

    def _get_llm_config_for_current_project(self) -> Dict[str, str]:
        """
        Повертає LLM-конфіг для поточного проєкту або дефолтні значення.
        """
        try:
            project_name = self._get_current_project_name()
            if project_name:
                return get_llm_config(project_name)
        except Exception:
            pass
        return {
            "base_url": env_default_base_url(),
            "head_model": os.getenv("HEAD_MODEL", "qwen2.5-7b-instruct-1m"),
        }

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

    def _is_pytest_request(self, user_text: str) -> bool:
        """Чи просить користувач запустити pytest."""
        t = (user_text or "").strip().lower()
        if not t:
            return False

        # Нормалізуємо початок (на випадок лапок/тире/пунктуації перед словом)
        norm = t.lstrip(" \t\n\r\"'`“”«»—–-\u00a0")

        # Підтримуємо найпоширеніші формулювання
        if "pytest" in norm and ("запусти" in norm or "run" in norm or norm.startswith("pytest")):
            return True

        return False

    def _truncate(self, s: str, limit: int = 6000) -> str:
        s = s or ""
        if len(s) <= limit:
            return s
        return s[:limit].rstrip() + "…"

    def _run_pytest(self) -> str:
        """Запускає pytest у поточному робочому каталозі репо."""
        # Найбільш сумісний виклик (не залежить від того, чи є pytest як окремий executable в PATH)
        cmd = ["python", "-m", "pytest", "-q"]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return "[pytest] Timeout (300s): тести виконуються занадто довго або зависли."
        except Exception as e:
            return f"[pytest] Не вдалося запустити pytest: {e}"

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()

        parts = ["[pytest] finished", f"return_code={proc.returncode}"]
        if out:
            parts.append("\n[stdout]\n" + self._truncate(out))
        if err:
            parts.append("\n[stderr]\n" + self._truncate(err))

        return "\n".join(parts)

    def _should_delegate(self, user_text: str) -> bool:
        """Евристика: коли варто делегувати задачу Supervisor-у.

        Ми делегуємо, коли запит схожий на "зроби дію в системі" (код/запуск/тести/гіт/аналіз/збірка).
        Не делегуємо для коротких розмовних відповідей.
        """
        debug = os.getenv("HEAD_DEBUG_DELEGATION", "").strip() == "1"

        def dbg(msg: str) -> None:
            if debug:
                print(f"[head:delegate?] {msg}")

        t = (user_text or "").strip()
        if not t:
            dbg("empty -> False")
            return False

        lower = t.lower()
        # Нормалізуємо початок (на випадок лапок/тире/пунктуації перед словом)
        norm = lower.lstrip(" \t\n\r\"'`“”«»—–-\u00a0")
        dbg(f"raw='{lower[:80]}' norm='{norm[:80]}'")

        # Явний opt-out
        if any(p in norm for p in ("не делегуй", "без делегування", "тільки поясни", "тільки відповідь")):
            dbg("opt-out phrase -> False")
            return False

        # Дуже короткі фрази зазвичай не потребують пайплайна
        if len(lower) <= 12 and lower in ("ок", "ok", "привіт", "hello", "дякую", "thanks"):
            dbg("very short greeting -> False")
            return False

        # Розмовні/"скажи"-запити зазвичай не потребують пайплайна Supervisor-а.
        conversational_prefixes = (
            "скажи",
            "скажіть",
            "say",
            "розкажи",
            "розкажіть",
            "поясни",
            "поясніть",
            "explain",
            "що таке",
            "що це",
            "what is",
        )
        if norm.startswith(conversational_prefixes):
            dbg("conversational prefix -> False")
            return False

        # Якщо користувач просить саме виконання / запуск / перевірку / зміни коду
        keywords = (
            # запуск/дія
            "запусти",
            "виконай",
            "зроби",
            "згенеруй",
            "створи",
            "перевір",
            "протестуй",
            "тест",
            "збірка",
            "build",
            "run",
            "execute",
            "pytest",
            # код/правки
            "код",
            "напиши код",
            "виправ",
            "пофіксь",
            "пофікси",
            "рефактор",
            "перепиши",
            "пул реквест",
            "pull request",
            # git
            "git ",
            "коміт",
            "commit",
            "push",
            "branch",
            "merge",
            # системні/cli
            "pip ",
            "python ",
            "curl ",
            "docker",
            # аналітика/лог/помилки
            "помилка",
            "errors",
            "traceback",
            "лог",
            "logs",
            "runs",
            "eval",
            "датасет",
            "dataset",
        )

        if any(k in norm for k in keywords):
            dbg("keyword hit -> True")
            return True

        # Якщо є явні маркери командного рядка
        if "```" in norm or norm.startswith("$"):
            dbg("command marker -> True")
            return True

        dbg("no rule matched -> False")
        return False

    def _format_supervisor_output(self, out: Any) -> str:
        """Нормалізуємо результат Supervisor-а до рядка."""
        if out is None:
            return ""

        if isinstance(out, str):
            return out.strip()

        # Часто це може бути dict з stdout/stderr/return_code або вкладеним результатом
        if isinstance(out, dict):
            # Найчастіші поля
            stdout = (
                out.get("stdout")
                or out.get("output")
                or out.get("text")
                or out.get("reply")
                or out.get("content")
                or out.get("result")
                or out.get("response")
                or out.get("final")
            )
            stderr = out.get("stderr")
            rc = out.get("return_code")

            # Інколи результат лежить всередині іншого dict
            for nested_key in ("data", "payload", "run", "meta", "message"):
                nested = out.get(nested_key)
                if isinstance(nested, dict):
                    stdout = stdout or (
                        nested.get("stdout")
                        or nested.get("output")
                        or nested.get("text")
                        or nested.get("reply")
                        or nested.get("content")
                        or nested.get("result")
                        or nested.get("response")
                        or nested.get("final")
                    )
                    stderr = stderr or nested.get("stderr")
                    if rc is None and nested.get("return_code") is not None:
                        rc = nested.get("return_code")

            parts: list[str] = []

            # Якщо є хоча б трохи контенту — повертаємо його
            if stdout:
                parts.append(str(stdout).strip())

            # Якщо контенту немає, але є технічні метадані — покажемо їх (краще ніж порожньо)
            if not parts:
                solver = out.get("solver") or out.get("solver_name")
                tags = out.get("tags")
                if solver or tags:
                    meta_bits = []
                    if solver:
                        meta_bits.append(f"solver={solver}")
                    if tags:
                        meta_bits.append(f"tags={tags}")
                    parts.append("[supervisor result: " + ", ".join(meta_bits) + "]")

            if stderr:
                parts.append("\n[stderr]\n" + str(stderr).strip())
            if rc is not None:
                parts.append(f"\n[return_code={rc}]")

            return "\n".join([p for p in parts if p and p.strip()])

        # Фолбек
        return str(out).strip()

    def _delegate_to_supervisor(self, task: str, memory: Any, *, auto: bool = True) -> Optional[str]:
        """Спроба виконати задачу через Supervisor.

        Реальний Supervisor у цьому проекті має метод `run`. Ми викликаємо його
        і підбираємо аргументи через inspect.signature(), щоб не ламатися при різних сигнатурах.

        Повертає текст відповіді або None, якщо делегування не вдалося.
        """
        debug = os.getenv("HEAD_DEBUG_DELEGATION", "").strip() == "1"

        def dbg(msg: str) -> None:
            if debug:
                print(f"[head:delegate] {msg}")

        fn = getattr(self.supervisor, "run", None)
        if not callable(fn):
            dbg("Supervisor.run is missing -> None")
            return None

        # Підбираємо kwargs під реальну сигнатуру
        kwargs: Dict[str, Any] = {}
        try:
            sig = inspect.signature(fn)
            param_names = set(sig.parameters.keys())
        except Exception as e:
            dbg(f"inspect.signature failed: {e}")
            param_names = set()

        if "memory" in param_names:
            kwargs["memory"] = memory

        # У більшості реалізацій Supervisor вже налаштований через auto_solver/auto_team,
        # але якщо у run() є прапор auto — передамо його.
        if "auto" in param_names:
            kwargs["auto"] = auto

        dbg(f"calling Supervisor.run(task, {kwargs})")

        out: Any
        try:
            out = fn(task, **kwargs)
        except TypeError as e:
            dbg(f"TypeError with kwargs: {e}; trying positional fallbacks")
            # Фолбеки на інші сигнатури
            try:
                out = fn(task, memory)
            except TypeError as e2:
                dbg(f"TypeError with (task, memory): {e2}; trying (task)")
                try:
                    out = fn(task)
                except Exception as e3:
                    dbg(f"Supervisor.run(task) failed: {e3}")
                    return None
            except Exception as e2:
                dbg(f"Supervisor.run(task, memory) failed: {e2}")
                return None
        except Exception as e:
            dbg(f"Supervisor.run failed: {e}")
            return None

        text = self._format_supervisor_output(out)
        if not text:
            if isinstance(out, dict):
                try:
                    dumped = json.dumps(out, ensure_ascii=False)
                except Exception:
                    dumped = repr(out)
                if len(dumped) > 600:
                    dumped = dumped[:600] + "…"
                dbg(f"Supervisor.run dict produced empty text; keys={list(out.keys())} dump={dumped}")
            else:
                dbg(f"Supervisor.run returned empty/falsey output: {type(out)}")
            return None

        dbg(f"delegation ok; len={len(text)}")
        return text

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
                lines.append(f"- id={p['id']} | name={p['name']} | type={p['type']} | status={p['status']}")
            return "\n".join(lines)

        if lower.startswith("переключись на проєкт") or lower.startswith("переключись на проект"):
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
                lines.append(f"[run_id={e['run_id']}] {e['error_type']}: {e['error_message']}")
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

        # --- 5. За замовчуванням: це звичайна "людська" задача → LLM як голова ---
        #
        # HeadAgent може АВТОМАТИЧНО делегувати деякі задачі Supervisor-у за простою евристикою.
        # --- 5a. Конкретні "дієві" команди, які краще виконати напряму ---
        if auto and self._is_pytest_request(clean):
            result = self._run_pytest()
            self._log_head_note(clean, result)
            return result

        # --- 5b. Загальна авто-делегація Supervisor-у ---
        if auto and self._should_delegate(clean):
            delegated = self._delegate_to_supervisor(clean, memory, auto=auto)
            if delegated:
                # Логуємо взаємодію так само, як і звичайну відповідь
                self._log_head_note(clean, delegated)
                return delegated

        reply = self.ask_llm(clean)
        self._log_head_note(clean, reply)
        return reply
