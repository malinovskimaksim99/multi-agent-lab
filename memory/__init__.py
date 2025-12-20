

"""Simple in-process memory for chat/head agent.

Right now це просто маленький сховище стану на час сесії.
Пізніше можемо розширити/замінити на більш складну реалізацію.
"""

from __future__ import annotations

from typing import Any, Dict


class Memory:
    """Проста пам'ять для однієї чат-сесії.

    Зараз це просто dict у процесі. Ми тримаємо тут тимчасові речі на кшталт:
    - поточний проєкт (current_project_id),
    - вибрана книга/глава/сцена для письменницьких задач,
    - проміжні нотатки сесії.

    Довгострокові речі (runs, head_notes тощо) ми все одно зберігаємо в БД через db.py.
    """

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def get(self, key: str, default: Any | None = None) -> Any | None:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Отримати повний словник (наприклад, для дебагу)."""
        return dict(self._data)


__all__ = ["Memory"]