from typing import Optional

from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent


@register_agent
class WriterAgent(BaseAgent):
    name = "writer"
    description = "Helps with writing, rewriting, outlines, and documentation."

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        t = task.lower()

        # Сильні маркери документації
        doc_markers = ["readme", "documentation", "doc", "guide", "installation"]
        if any(m in t for m in doc_markers):
            return 0.95

        keywords = [
            "write", "rewrite", "story", "outline", "essay",
            "text", "script", "email", "post", "article",
        ]
        hits = sum(1 for k in keywords if k in t)
        if hits == 0:
            return 0.1
        return min(1.0, 0.5 + hits * 0.15)

    @staticmethod
    def _is_install_task(t: str) -> bool:
        return any(k in t for k in ["installation", "install", "setup", "встановлен"])

    @staticmethod
    def _is_readme_outline(t: str) -> bool:
        return "readme" in t and any(k in t for k in ["outline", "структур", "мінімал"])

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        t = task.lower()

        # 1) Кейс: коротка README-секція для installation
        if self._is_install_task(t):
            lines = [
                "## Встановлення",
                "",
                "### Попередні вимоги",
                "- Встановлений Python 3.10+.",
                "- Встановлений git.",
                "",
                "### Кроки",
                "1. Клонувати репозиторій:",
                "   git clone <URL_РЕПОЗИТОРІЮ>",
                "   cd <КАТАЛОГ_ПРОЄКТУ>",
                "2. Створити віртуальне середовище:",
                "   python -m venv .venv",
                "3. Активувати віртуальне середовище:",
                "   source .venv/bin/activate  (Linux/macOS)",
                "   .venv\\Scripts\\activate   (Windows)",
                "4. Встановити залежності:",
                "   pip install -r requirements.txt",
                "5. За потреби налаштувати змінні оточення або файл конфігурації.",
                "6. Перевірити запуск програми:",
                "   python app.py --help",
            ]
            output = "\n".join(lines)
            return AgentResult(agent=self.name, output=output, meta={"mode": "readme_install"})

        # 2) Кейс: мінімальний каркас README
        if self._is_readme_outline(t):
            lines = [
                "## Мінімальний каркас README",
                "",
                "- Назва проєкту та короткий опис.",
                "- Розділ «Встановлення».",
                "- Розділ «Використання».",
                "- Налаштування / змінні оточення (якщо є).",
                "- Приклади команд.",
                "- Як запускати тести (якщо є).",
                "- Ліцензія та контакти / посилання.",
            ]
            output = "\n".join(lines)
            return AgentResult(agent=self.name, output=output, meta={"mode": "readme_outline"})

        # 3) Інші writing-задачі — простий структурований чернетковий текст
        lines = [
            "## Завдання",
            task,
            "",
            "### Огляд",
            "Це задача на написання тексту, потрібна чітка та читабельна відповідь.",
            "",
            "### Ключові пункти",
            "- Сформулюйте мету в 1–2 реченнях.",
            "- Дайте коротке пояснення з кількома конкретними деталями.",
            "- Завершіть коротким висновком або наступним кроком.",
        ]
        output = "\n".join(lines)

        return AgentResult(agent=self.name, output=output, meta={"mode": "content"})
