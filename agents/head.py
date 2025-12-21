# agents/head.py
from __future__ import annotations

import os
import inspect
import json
import subprocess
import re

from typing import Any, Dict, Optional

import repo_tools as rt
try:
    import tools_allowlist as ta
except Exception:
    ta = None  # type: ignore

from .supervisor import Supervisor

from db import (
    get_projects,
    get_current_project,
    set_current_project,
    get_recent_errors,  # ми вже додавали це раніше
    add_head_note,
    get_llm_config,
    get_project_id_by_name,
    get_head_notes_by_project_id,
    delete_head_notes,
    dedupe_head_notes,
    search_head_notes,
    delete_head_note_by_id,
)

from .head_profile import build_head_system_prompt
from llm_client import chat_openai_compat, env_default_base_url


class HeadAgent:
    def _looks_like_preference(self, lower_norm: str) -> bool:
        """М'яке побажання/скарга без маркерів домовленості — відповісти коротко, без сейву."""
        s = (lower_norm or "").strip()
        if not s:
            return False
        # якщо це питання — точно не preference-shortcut
        if "?" in s:
            return False
        markers = (
            "не треба щоб",
            "не потрібно щоб",
            "не хочу щоб",
            "мені не треба",
            "будь ласка не",
            "не роби",
            "не робіть",
            "не роби так",
        )
        return any(m in s for m in markers) and len(s) <= 180
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
            project = None
            try:
                project = get_current_project()
            except Exception:
                project = None
            if isinstance(project, dict):
                project = project.get("name")
            if isinstance(project, str):
                project = project.strip()
            else:
                project = None

            cfg: Dict[str, str] = {}
            if project:
                try:
                    cfg = get_llm_config(project)
                except Exception:
                    cfg = {}

            base_url = cfg.get("base_url") or env_default_base_url()
            model = cfg.get("head_model") or os.getenv(
                "HEAD_MODEL",
                "qwen2.5-7b-instruct-1m",
            )
            notes_block = self._build_notes_context()
            system_prompt = self.system_prompt
            if notes_block:
                system_prompt = system_prompt + "\n\n" + notes_block
            reply = chat_openai_compat(
                base_url=base_url,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
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
        if not self._interaction_logging_enabled():
            return
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
                kind="interaction",
            )
            return
        except TypeError:
            # Можливо, функція приймає інші імена параметрів/менше аргументів.
            pass
        except Exception:
            return

    def _shorten_text(self, text: str, limit: int = 220) -> str:
        s = " ".join((text or "").split())
        if len(s) <= limit:
            return s
        return s[:limit].rstrip() + "…"

    def _needs_follow_up(self, text: str) -> bool:
        t = (text or "").lower()
        if not t:
            return False
        markers = ["todo", "next", "follow", "потрібно", "далі", "наступ", "?"]
        return any(m in t for m in markers)

    def log_writer_shadow(self, user_text: str, writer_reply: str) -> None:
        """Записує короткий shadow-лог WriterAgent-а в head_notes."""
        if not self._shadow_logging_enabled():
            return
        project_name = self._get_current_project_name()
        if not project_name:
            return

        short_task = self._shorten_text(user_text, 180)
        short_reply = self._shorten_text(writer_reply, 240)
        next_action = (
            "Потрібні подальші дії."
            if self._needs_follow_up(writer_reply)
            else "Подальші дії не потрібні."
        )
        note_text = (
            "SHADOW (writer): "
            f"Запит: {short_task}. "
            f"Відповідь: {short_reply}. "
            f"{next_action}"
        )

        try:
            add_head_note(
                project_name,
                scope="project",
                note_type="shadow",
                note=note_text,
                importance=1,
                tags="shadow,writer",
                source="writer_mode",
                author="head",
                kind="shadow",
            )
            return
        except TypeError:
            pass
        except Exception:
            return

        try:
            add_head_note(project_name, note_text, kind="shadow")
        except Exception:
            return

        try:
            # Мінімальний фолбек: тільки те, що майже напевно є.
            add_head_note(project_name, note_text, kind="shadow")
        except Exception:
            return

    def _format_head_notes(self, notes: Any, empty_message: str) -> str:
        """Форматує список нотаток у короткий перелік."""
        if not notes:
            return empty_message

        # Очікуємо list[dict], але робимо фолбеки
        if not isinstance(notes, list):
            return str(notes)

        lines = []
        for n in notes:
            if isinstance(n, dict):
                created = (n.get("created_at") or n.get("updated_at") or "").strip()
                kind = (n.get("kind") or n.get("note_type") or "").strip()
                note = (n.get("note") or "").strip()
                if len(note) > 260:
                    note = note[:260].rstrip() + "…"
                meta = " | ".join([p for p in [created, kind] if p])
                if meta:
                    lines.append(f"- [{meta}] {note}")
                else:
                    lines.append(f"- {note}")
            else:
                lines.append(f"- {n}")
        return "\n".join(lines)

    def _format_note_items(self, notes: Any, empty_message: str) -> str:
        """Форматує список корисних нотаток з тегами та id."""
        if not notes:
            return empty_message

        if not isinstance(notes, list):
            return str(notes)

        lines = []
        for n in notes:
            if isinstance(n, dict):
                created = (n.get("created_at") or n.get("updated_at") or "").strip()
                tag = (n.get("tags") or n.get("note_type") or "note").strip()
                note_id = n.get("id")
                note = (n.get("note") or "").strip()
                if len(note) > 260:
                    note = note[:260].rstrip() + "…"
                meta_parts = []
                if created:
                    meta_parts.append(created)
                if tag:
                    meta_parts.append(tag)
                if note_id is not None:
                    meta_parts.append(f"id={note_id}")
                meta = " | ".join(meta_parts)
                if meta:
                    lines.append(f"- [{meta}] {note}")
                else:
                    lines.append(f"- {note}")
            else:
                lines.append(f"- {n}")
        return "\n".join(lines)

    def _extract_notes_limit(self, text: str, default: int = 10) -> int:
        """Витягує ліміт з тексту запиту (українською/англійською)."""
        matches = re.findall(r"\d+", text)
        if matches:
            try:
                value = int(matches[0])
                if value > 0:
                    return value
            except Exception:
                pass
        # Якщо явно просять "всі", даємо підвищений ліміт.
        if "всі" in text or "all" in text:
            return 200
        return default

    def _interaction_logging_enabled(self) -> bool:
        """Чи дозволено логувати interaction-нотатки (за замовчуванням вимкнено)."""
        value = os.getenv("HEAD_LOG_INTERACTIONS", "").strip().lower()
        return value in ("1", "true", "yes", "on")

    def _shadow_logging_enabled(self) -> bool:
        """Чи дозволено логувати shadow-нотатки (за замовчуванням вимкнено)."""
        value = os.getenv("HEAD_LOG_SHADOW", "").strip().lower()
        return value in ("1", "true", "yes", "on")

    def _get_current_project_id(self) -> Optional[int]:
        """Повертає project_id для поточного проєкту."""
        project_name = self._get_current_project_name()
        if not project_name:
            return None
        try:
            return get_project_id_by_name(project_name)
        except Exception:
            return None

    def _normalize_note_text(self, text: str) -> str:
        """Нормалізує текст нотатки для дедупу."""
        s = (text or "").strip()
        for ch in ("’", "ʼ", "`"):
            s = s.replace(ch, "'")
        lower = s.lower()
        prefixes = (
            "запам'ятай правило",
            "запам'ятай",
            "правило",
            "rule",
            "нотатка",
            "note",
            "збережи",
        )
        for p in prefixes:
            if lower.startswith(p):
                s = s[len(p):].strip()
                break
        s = s.lstrip(":—- ").strip()
        if len(s) >= 2:
            quote_pairs = (
                ('"', '"'),
                ("'", "'"),
                ("“", "”"),
                ("«", "»"),
            )
            for left, right in quote_pairs:
                if s.startswith(left) and s.endswith(right):
                    s = s[1:-1].strip()
                    break
        s = re.sub(r"\s+", " ", s)
        return s

    def _parse_note_save_request(self, lower_norm: str) -> Optional[str]:
        """Повертає тег для запиту 'запамʼятай ...', якщо це він."""
        if lower_norm.startswith("запам'ятай правило:"):
            return "rule"
        if lower_norm.startswith("запам'ятай рішення:"):
            return "decision"
        if lower_norm.startswith("запам'ятай факт:"):
            return "fact"
        if lower_norm.startswith("запам'ятай:"):
            return "note"
        if lower_norm.startswith("правило:"):
            return "rule"
        if lower_norm.startswith("rule:"):
            return "rule"
        if lower_norm.startswith("збережи:"):
            return "note"
        if lower_norm.startswith("нотатка:"):
            return "note"
        if lower_norm.startswith("note:"):
            return "note"
        return None

    def _build_notes_context(self) -> str:
        """Готує блок корисних нотаток для LLM."""
        project_id = self._get_current_project_id()
        if not project_id:
            return ""
        try:
            notes = get_head_notes_by_project_id(
                project_id,
                kinds=["note", "rule"],
                limit=10,
            )
        except Exception:
            return ""
        if not notes:
            return ""
        lines = ["Корисні нотатки (поточний проєкт):"]
        for n in notes:
            tag = (n.get("tags") or n.get("note_type") or "note").strip()
            text = (n.get("note") or "").strip()
            if len(text) > 220:
                text = text[:220].rstrip() + "…"
            lines.append(f"- [{tag}] {text}")
        return "\n".join(lines)

    def _save_note(self, tag: str, note_text: str) -> str:
        """Зберігає нотатку з дедупом."""
        project_name = self._get_current_project_name()
        if not project_name:
            return "Немає активного проєкту, тому не можу зберегти нотатку."

        note_kind = "rule" if tag == "rule" else "note"

        project_id = self._get_current_project_id()
        if project_id:
            try:
                existing = get_head_notes_by_project_id(
                    project_id,
                    kinds=[note_kind],
                    limit=200,
                )
            except Exception:
                existing = []
            for n in existing:
                existing_text = self._normalize_note_text(str(n.get("note", "")))
                existing_kind = (n.get("kind") or "").strip() or "note"
                existing_tag = (n.get("tags") or n.get("note_type") or "").strip()
                if (
                    existing_kind == note_kind
                    and existing_text == note_text
                    and existing_tag == tag
                ):
                    return "Вже є така нотатка."

        try:
            add_head_note(
                project_name,
                scope="project",
                note_type=tag,
                note=note_text,
                importance=1,
                tags=tag,
                source="user",
                author="user",
                kind=note_kind,
            )
        except Exception as e:
            return f"Не вдалося зберегти нотатку: {e}"

        return "Збережено."

    def _get_flag(self, memory: Any, name: str, default: Any = None) -> Any:
        """Безпечне читання прапорця з memory."""
        if memory is None:
            return default
        getter = getattr(memory, "get_flag", None)
        if callable(getter):
            return getter(name, default)
        flags = getattr(memory, "flags", None)
        if isinstance(flags, dict):
            return flags.get(name, default)
        return default

    def _set_flag(self, memory: Any, name: str, value: Any) -> None:
        """Безпечне встановлення прапорця у memory."""
        if memory is None:
            return
        setter = getattr(memory, "set_flag", None)
        if callable(setter):
            setter(name, value)
            return
        flags = getattr(memory, "flags", None)
        if isinstance(flags, dict):
            flags[name] = value

    def _clear_flag(self, memory: Any, name: str) -> None:
        """Безпечне очищення прапорця у memory."""
        if memory is None:
            return
        setter = getattr(memory, "set_flag", None)
        if callable(setter):
            setter(name, None)
            return
        flags = getattr(memory, "flags", None)
        if isinstance(flags, dict):
            flags.pop(name, None)

    def _set_pending_queue(self, memory: Any, queue: list[dict]) -> None:
        """Записує pending-чергу в memory."""
        if memory is None:
            return
        flags = getattr(memory, "flags", None)
        if not isinstance(flags, dict):
            try:
                setattr(memory, "flags", {})
                flags = memory.flags  # type: ignore[attr-defined]
            except Exception:
                flags = None
        if isinstance(flags, dict):
            flags["pending_memory"] = queue
            return
        self._set_flag(memory, "pending_memory", queue)

    def _get_pending_queue(self, memory: Any) -> list[dict]:
        """Повертає pending-чергу (FIFO)."""
        pending = None
        flags = getattr(memory, "flags", None)
        if isinstance(flags, dict):
            pending = flags.get("pending_memory")
        if pending is None:
            pending = self._get_flag(memory, "pending_memory")

        if isinstance(pending, dict):
            pending = [pending]

        queue: list[dict] = []
        if isinstance(pending, list):
            for item in pending:
                if isinstance(item, dict) and item.get("kind") and item.get("text"):
                    queue.append(
                        {"kind": str(item["kind"]), "text": str(item["text"])}
                    )
            # Нормалізуємо зворотно (list замість dict, без сміття).
            self._set_pending_queue(memory, queue)
        return queue

    def _enqueue_pending(self, memory: Any, kind: str, text: str) -> int:
        """Додає pending у кінець черги (без дублів)."""
        queue = self._get_pending_queue(memory)
        for idx, item in enumerate(queue, start=1):
            if item.get("kind") == kind and item.get("text") == text:
                return idx
        queue.append({"kind": kind, "text": text})
        self._set_pending_queue(memory, queue)
        return len(queue)

    def _pop_pending(self, memory: Any) -> Optional[Dict[str, str]]:
        """Знімає перший pending з черги."""
        queue = self._get_pending_queue(memory)
        if not queue:
            return None
        item = queue.pop(0)
        if queue:
            self._set_pending_queue(memory, queue)
        else:
            self._clear_flag(memory, "pending_memory")
        return item

    def _peek_pending(self, memory: Any) -> Optional[Dict[str, str]]:
        """Повертає перший pending без змін."""
        queue = self._get_pending_queue(memory)
        if not queue:
            return None
        return queue[0]

    def _pending_count(self, memory: Any) -> int:
        """Кількість pending-елементів у черзі."""
        return len(self._get_pending_queue(memory))

    def _is_yes(self, text: str) -> bool:
        """Чи є відповідь підтвердженням."""
        t = (text or "").strip().lower()
        t = t.strip(" .,!?:;")
        return t in ("так", "yes", "ok", "ок")

    def _is_no(self, text: str) -> bool:
        """Чи є відповідь відмовою."""
        t = (text or "").strip().lower()
        t = t.strip(" .,!?:;")
        return t in ("ні", "no", "не треба", "не потрібно", "скасувати", "cancel")

    def _should_prompt_rule(self, lower_norm: str) -> bool:
        """Чи варто перепитати про правило."""
        if lower_norm.startswith(("нотатки", "лог", "покажи", "видали", "почисти", "tool ")):
            return False
        if "?" in lower_norm:
            return False
        if any(
            m in lower_norm
            for m in ("може", "напевно", "як думаєш", "здається")
        ):
            return False
        if re.search(r"\bчи\b", lower_norm):
            return False
        s = lower_norm.strip()
        # Питаємо ТІЛЬКИ при явних маркерах “домовленості”
        return s.startswith(
            (
                "відтепер",
                "завжди",
                "ніколи",
                "правило",
                "rule",
                "давай домовимось",
                "робимо так",
            )
        )

    def _is_notes_view_request(self, lower_text: str) -> bool:
        """Чи просить користувач показати нотатки."""
        t = (lower_text or "").strip()
        if t.startswith("нотатки пошук"):
            return False

        # Командні форми (строго), щоб не тригеритись на "нотаткою/нотатка" в звичайних реченнях
        if re.match(r"^(нотатки)(\s+\d+)?$", t):
            return True
        if re.match(r"^(покажи\s+нотатки)(\s+\d+)?$", t):
            return True
        if t in (
            "head нотатки",
            "покажи head нотатки",
            "покажи нотатки head",
            "show head notes",
        ):
            return True
        return False

    def _is_log_view_request(self, lower_text: str) -> bool:
        """Чи просить користувач показати лог."""
        if re.search(r"\bлог\b", lower_text):
            return True
        if "show log" in lower_text:
            return True
        return False

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
        lower_norm = lower.replace("’", "'").replace("ʼ", "'").replace("`", "'")

        # --- 0. Pending-підтвердження (до будь-яких LLM/route) ---
        pending_view = lower_norm.startswith(
            ("pending", "непідтверджені", "покажи pending", "покажи непідтверджені")
        )
        pending_queue = self._get_pending_queue(memory)
        if pending_queue and not pending_view:
            if self._is_yes(lower):
                item = self._pop_pending(memory)
                if item:
                    result = self._save_note(item["kind"], item["text"])
                    if result.startswith("Не вдалося"):
                        return result
                if self._pending_count(memory) > 0:
                    next_item = self._peek_pending(memory)
                    if next_item:
                        preview = self._shorten_text(next_item["text"], 120)
                        return f"Ок. Наступне правило: {preview}. Зберегти? (так/ні)"
                return "Збережено."
            if self._is_no(lower):
                self._pop_pending(memory)
                if self._pending_count(memory) > 0:
                    next_item = self._peek_pending(memory)
                    if next_item:
                        preview = self._shorten_text(next_item["text"], 120)
                        return f"Ок. Наступне правило: {preview}. Зберегти? (так/ні)"
                return "Ок, не зберігаю."
            # якщо користувач продовжив іншою думкою — можемо додати правило в чергу
            if self._should_prompt_rule(lower_norm):
                note_text = self._normalize_note_text(clean)
                if note_text:
                    pos = self._enqueue_pending(memory, "rule", note_text)
                    preview = self._shorten_text(note_text, 120)
                    return (
                        f"Додав у чергу правило #{pos}: {preview}. "
                        "Зберегти? (так/ні). Подивитись список: 'pending'."
                    )
            return "Підтверди 'так' або 'ні'."

        # --- 1. Перегляд pending-черги (до будь-яких LLM/route) ---
        if pending_view:
            queue = self._get_pending_queue(memory)
            if not queue:
                return "Немає непідтверджених правил."
            lines = ["Непідтверджені правила:"]
            for idx, item in enumerate(queue, start=1):
                preview = self._shorten_text(item["text"], 120)
                lines.append(f"{idx}) {preview}")
            return "\n".join(lines)

        # --- 2. Класифікація memory intent (до будь-яких LLM/route) ---
        tag = self._parse_note_save_request(lower_norm)
        if tag:
            parts = clean.split(":", 1)
            note_text = parts[1].strip() if len(parts) > 1 else ""
            note_text = self._normalize_note_text(note_text)
            if not note_text:
                return "ОК, що саме запамʼятати?"
            return self._save_note(tag, note_text)

        if self._should_prompt_rule(lower_norm):
            note_text = self._normalize_note_text(clean)
            if note_text:
                pos = self._enqueue_pending(memory, "rule", note_text)
                preview = self._shorten_text(note_text, 120)
                return (
                    f"Додав у чергу правило #{pos}: {preview}. "
                    "Зберегти? (так/ні). Подивитись список: 'pending'."
                )

        # Побажання без маркерів домовленості — просто коротко підтвердити (без логів/нотаток)
        if self._looks_like_preference(lower_norm):
            return "Ок, прийняв."

        # --- 0. Tools-first allowlist (explicit tool invocation) ---
        if lower.startswith("tool "):
            tool_text = clean[5:].strip()
            if not tool_text:
                result = "Usage: tool <name> [args]"
                self._log_head_note(clean, result)
                return result
            if ta is None:
                result = "[tool] error: tools_allowlist not available"
                self._log_head_note(clean, result)
                return result

            name, *rest = tool_text.split(" ", 1)
            args: Dict[str, Any] = {}
            if rest:
                rest_str = rest[0].strip()
                if rest_str:
                    if rest_str.startswith("{") and rest_str.endswith("}"):
                        try:
                            parsed = json.loads(rest_str)
                            if isinstance(parsed, dict):
                                args = parsed
                            else:
                                result = "Tool args must be a JSON object"
                                self._log_head_note(clean, result)
                                return result
                        except Exception as e:
                            result = f"Invalid JSON args: {e}"
                            self._log_head_note(clean, result)
                            return result
                    else:
                        if name == "repo_search":
                            args = {"query": rest_str}
                        elif name == "git_diff" and rest_str.isdigit():
                            args = {"limit": int(rest_str)}
                        elif name == "recent_errors" and rest_str.isdigit():
                            args = {"limit": int(rest_str)}
                        else:
                            result = "Unsupported tool args format"
                            self._log_head_note(clean, result)
                            return result

            try:
                out = ta.run_tool(name, args)
                if out.get("ok"):
                    payload = out.get("data")
                    dumped = json.dumps(payload, ensure_ascii=False, indent=2)
                    result = "[tool] ok\n" + self._truncate(dumped)
                else:
                    result = "[tool] error: " + str(out.get("error", "unknown error"))
            except Exception as e:
                result = "[tool] error: " + str(e)

            self._log_head_note(clean, result)
            return result

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

        # --- 4. Нотатки і лог HeadAgent-а ---
        if lower.startswith("почисти старий лог") or lower.startswith("почисти лог"):
            project_id = self._get_current_project_id()
            if not project_id:
                return "Немає активного проєкту, тому лог чистити не можу."
            try:
                deleted = delete_head_notes(project_id, kinds=["interaction", "shadow"])
            except Exception as e:
                return f"Не вдалося почистити лог: {e}"
            return f"Лог очищено ({deleted} записів)."

        if lower.startswith("почисти нотатки"):
            project_id = self._get_current_project_id()
            if not project_id:
                return "Немає активного проєкту, тому нотатки чистити не можу."
            try:
                deleted = delete_head_notes(project_id, kinds=["note"])
            except Exception as e:
                return f"Не вдалося почистити нотатки: {e}"
            return f"Нотатки очищено ({deleted} записів)."

        if lower.startswith("почисти все"):
            if "підтверджую" not in lower:
                return "Щоб почистити все, напиши: 'почисти все підтверджую'."
            project_id = self._get_current_project_id()
            if not project_id:
                return "Немає активного проєкту, тому чистити нічого."
            try:
                deleted = delete_head_notes(project_id)
            except Exception as e:
                return f"Не вдалося почистити все: {e}"
            return f"Усі нотатки очищено ({deleted} записів)."

        if lower.startswith("прибери дублікати нотаток"):
            project_id = self._get_current_project_id()
            if not project_id:
                return "Немає активного проєкту, тому дублікати прибрати не можу."
            try:
                removed = dedupe_head_notes(project_id)
            except Exception as e:
                return f"Не вдалося прибрати дублікати: {e}"
            return f"Дублікати прибрано ({removed} записів)."

        if lower.startswith("видали нотатку"):
            project_id = self._get_current_project_id()
            if not project_id:
                return "Немає активного проєкту, тому нотатку видалити не можу."
            match = re.search(r"\b(\d+)\b", lower)
            if not match:
                return "Вкажи id нотатки, наприклад: 'видали нотатку 12'."
            try:
                note_id = int(match.group(1))
            except Exception:
                return "Невірний id нотатки."
            try:
                ok = delete_head_note_by_id(project_id, note_id)
            except Exception as e:
                return f"Не вдалося видалити нотатку: {e}"
            if not ok:
                return "Нотатку не знайдено."
            return "Нотатку видалено."

        if lower.startswith("нотатки пошук"):
            project_id = self._get_current_project_id()
            if not project_id:
                return "Немає активного проєкту, тому нотатки показати не можу."
            search_text = clean[len("нотатки пошук"):].strip()
            if not search_text:
                return "Вкажи текст для пошуку, наприклад: 'нотатки пошук логувати'."
            try:
                notes = search_head_notes(project_id, search_text, limit=20)
            except Exception as e:
                return f"Не вдалося виконати пошук: {e}"
            return self._format_note_items(notes, "Нічого не знайдено.")

        if self._is_log_view_request(lower):
            project_id = self._get_current_project_id()
            if not project_id:
                return "Немає активного проєкту, тому лог показати не можу."
            try:
                limit = self._extract_notes_limit(lower, default=20)
                notes = get_head_notes_by_project_id(
                    project_id,
                    kinds=["interaction", "shadow"],
                    limit=limit,
                )
            except Exception as e:
                return f"Не вдалося прочитати лог: {e}"
            return self._format_head_notes(notes, "Лог порожній.")

        if self._is_notes_view_request(lower):
            project_id = self._get_current_project_id()
            if not project_id:
                return "Немає активного проєкту, тому нотатки показати не можу."
            try:
                limit = self._extract_notes_limit(lower, default=10)
                notes = get_head_notes_by_project_id(
                    project_id,
                    kinds=["note"],
                    limit=limit,
                )
            except Exception as e:
                return f"Не вдалося прочитати нотатки: {e}"
            return self._format_note_items(notes, "Нотаток HeadAgent-а поки немає.")

        # --- 5. Детерміновані репо-інструменти (без LLM) ---
        if any(
            p in lower
            for p in (
                "git status",
                "покажи статус",
                "статус репо",
                "статус репозиторію",
                "перевір статус репозиторію",
            )
        ):
            try:
                data = rt.git_status()
                parts = ["[git] git status --porcelain=v1 -b", "return_code=0"]
                if data.get("stdout"):
                    parts.append("\n[stdout]\n" + data["stdout"])
                if data.get("stderr"):
                    parts.append("\n[stderr]\n" + data["stderr"])
                result = "\n".join(parts)
            except Exception as e:
                result = (
                    "[git] git status --porcelain=v1 -b\n"
                    "return_code=error\n"
                    f"[stderr]\n{e}"
                )
            self._log_head_note(clean, result)
            return result

        if any(
            p in lower
            for p in (
                "git diff",
                "покажи diff",
                "покажи зміни",
            )
        ):
            try:
                data = rt.git_diff()
                if data.get("truncated"):
                    stat_parts = ["[git] git diff --stat", "return_code=0"]
                    if data.get("stat"):
                        stat_parts.append("\n[stdout]\n" + data["stat"])
                    if data.get("stderr"):
                        stat_parts.append("\n[stderr]\n" + data["stderr"])
                    result = "diff truncated, showing --stat\n" + "\n".join(stat_parts)
                else:
                    parts = ["[git] git diff", "return_code=0"]
                    if data.get("diff"):
                        parts.append("\n[stdout]\n" + data["diff"])
                    if data.get("stderr"):
                        parts.append("\n[stderr]\n" + data["stderr"])
                    result = "\n".join(parts)
            except Exception as e:
                result = "[git] git diff\nreturn_code=error\n[stderr]\n" + str(e)
            self._log_head_note(clean, result)
            return result

        search_query = None
        if lower.startswith("grep "):
            search_query = clean[5:].strip()
        elif lower.startswith("знайди в репо"):
            search_query = clean[len("знайди в репо") :].strip(" :")
        elif lower.startswith("пошукай в репо"):
            search_query = clean[len("пошукай в репо") :].strip(" :")
        elif lower.startswith("пошук в репо"):
            search_query = clean[len("пошук в репо") :].strip(" :")

        if search_query is not None:
            if len(search_query) < 2:
                result = "Підкажи, що шукати, наприклад: 'знайди в репо head.py'."
                self._log_head_note(clean, result)
                return result
            try:
                data = rt.repo_search(search_query)
                if not data.get("found"):
                    result = "нічого не знайдено"
                else:
                    parts = [f"[grep] {search_query}", "return_code=0"]
                    if data.get("matches"):
                        parts.append("\n[stdout]\n" + data["matches"])
                    if data.get("stderr"):
                        parts.append("\n[stderr]\n" + data["stderr"])
                    result = "\n".join(parts)
            except Exception as e:
                result = f"[grep] Не вдалося виконати пошук: {e}"
            self._log_head_note(clean, result)
            return result

        # --- 6. За замовчуванням: це звичайна "людська" задача → LLM як голова ---
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
