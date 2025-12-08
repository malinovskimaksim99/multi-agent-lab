from typing import Optional

from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent


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
            output = "\n".join(lines)
            return AgentResult(agent=self.name, output=output, meta={"mode": "debugging"})

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
            output = "\n".join(lines)
            return AgentResult(agent=self.name, output=output, meta={"mode": "freecad_macro"})

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
        output = "\n".join(lines)
        return AgentResult(agent=self.name, output=output, meta={"mode": "generic_code"})