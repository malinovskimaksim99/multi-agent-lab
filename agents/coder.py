from typing import Optional, Dict, Any, List

import sqlite3
import json
from pathlib import Path

from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent


def _load_agent_config(agent_name: str) -> Dict[str, Any]:
    """
    Зчитує конфіг для конкретного агента з таблиці agent_configs у runs.db.
    Повертає dict {config_key: parsed_value}.
    Якщо БД або таблиця відсутні, повертає порожній словник.
    """
    db_path = Path("runs.db")
    if not db_path.exists():
        return {}

    cfg: Dict[str, Any] = {}
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            "SELECT config_key, config_value FROM agent_configs WHERE agent_name = ?",
            (agent_name,),
        )
        rows = cur.fetchall()
    except Exception:
        return {}
    finally:
        try:
            conn.close()
        except Exception:
            pass

    for key, value in rows:
        parsed: Any = value
        if isinstance(value, str):
            # пробуємо розпарсити JSON (для списків тощо)
            try:
                parsed = json.loads(value)
            except Exception:
                parsed = value
        cfg[key] = parsed

    return cfg


def _apply_style(lines: List[str], config: Dict[str, Any]) -> List[str]:
    """
    Легка стилізація виводу на основі конфігу агента.

    Підтримувані ключі:
      - style: "compact" або "detailed"
      - max_lines: int, максимальна кількість рядків у відповіді
    """
    if not config:
        return lines

    style = config.get("style")
    max_lines = config.get("max_lines")

    out = list(lines)

    # Якщо задано max_lines — просто обрізаємо відповідь
    if isinstance(max_lines, int) and max_lines > 0:
        out = out[:max_lines]

    # Стилі
    if style == "compact":
        # Прибираємо зайві порожні рядки
        out = [ln for ln in out if ln.strip()]
    elif style == "detailed":
        # Якщо немає явного підсумку — додаємо невеликий блок
        has_summary = any("Висновок" in ln or "Підсумок" in ln for ln in out)
        if not has_summary:
            out = out + [
                "",
                "### Підсумок",
                "- Підсумуйте головну ідею власними словами.",
            ]

    return out


@register_agent
class CoderAgent(BaseAgent):
    name = "coder"
    description = "Helps with code-related tasks: Python snippets, debugging, simple scripts."

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        t = task.lower()

        # Сильні маркери помилок / traceback
        if "traceback" in t or "error" in t or "exception" in t:
            return 0.9

        # Python / скрипти / макроси
        code_markers = [
            "python",
            ".py",
            "script",
            "скрипт",
            "код",
            "function",
            "class",
            "macro",
            "макрос",
            "freecad",
        ]
        hits = sum(1 for k in code_markers if k in t)
        if hits == 0:
            return 0.1

        return min(1.0, 0.6 + hits * 0.1)

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        t = task.lower()
        config = _load_agent_config(self.name)

        # 1) Режим: розбір помилки / traceback
        if "traceback" in t or "error" in t or "exception" in t:
            lines = [
                "## Аналіз помилки в коді",
                "",
                "### Кроки розбору",
                "1. Знайдіть у traceback останній блок з текстом помилки (тип, файл, рядок).",
                "2. Зверніть увагу на тип помилки (наприклад, `SyntaxError`, `TypeError`, `ImportError`).",
                "3. Подивіться, на який рядок у файлі посилається traceback.",
                "4. Проаналізуйте, що саме не так у цьому місці (неправильний виклик, відсутній модуль, помилковий тип тощо).",
                "",
                "### Що зазвичай робити далі",
                "- Прочитати уважно текст помилки англійською — він часто напряму підказує суть.",
                "- Звірити версію Python / бібліотек, якщо йдеться про `ImportError` або `AttributeError`.",
                "- Спростити підозрілий фрагмент коду до мінімального прикладу і перевірити його окремо.",
            ]
            lines = _apply_style(lines, config)
            output = "\n".join(lines)
            return AgentResult(
                agent=self.name,
                output=output,
                meta={"mode": "debugging", "config": config},
            )

        # 2) Режим: згадка FreeCAD / макросів
        if "freecad" in t or "macro" in t or "макрос" in t:
            lines = [
                "## Каркас Python-макроса для FreeCAD",
                "",
                "### Основні кроки",
                "1. Імпортувати модулі FreeCAD та FreeCADGui.",
                "2. Отримати активний документ або створити новий.",
                "3. Створити потрібні обʼєкти (бокси, циліндри, ескізи тощо).",
                "4. Налаштувати параметри обʼєктів (розміри, позиції, повороти).",
                "5. Оновити документ (`recompute`) і зберегти при потребі.",
                "",
                "### Що ще варто врахувати",
                "- Вказати одиниці виміру (мм, см тощо).",
                "- Додати просту обробку помилок, якщо документ відсутній.",
                "- Зафіксувати типові параметри у змінних на початку скрипта.",
            ]
            lines = _apply_style(lines, config)
            output = "\n".join(lines)
            return AgentResult(
                agent=self.name,
                output=output,
                meta={"mode": "freecad_macro", "config": config},
            )

        # 3) Загальний режим: задача на написання коду / скрипта
        lines = [
            "## План для задачі з кодом",
            "",
            "### 1. Уточнення задачі",
            "- Сформулюйте, що саме має робити скрипт (вхідні дані, вихід, обмеження).",
            "",
            "### 2. Структура рішення",
            "- Визначте основні кроки алгоритму у вигляді списку.",
            "- Подумайте, які функції логічно виділити окремо.",
            "",
            "### 3. Реалізація",
            "- Створіть порожній файл з кодом і додайте мінімальний каркас (точка входу).",
            "- Додавайте реалізацію кожного кроку по черзі, тестуючи після змін.",
            "",
            "### 4. Перевірка",
            "- Перевірте кілька сценаріїв: нормальний випадок, порожні дані, некоректні значення.",
            "- Додайте короткі коментарі там, де логіка неочевидна.",
        ]
        lines = _apply_style(lines, config)
        output = "\n".join(lines)
        return AgentResult(
            agent=self.name,
            output=output,
            meta={"mode": "generic_code", "config": config},
        )