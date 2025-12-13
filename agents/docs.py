from typing import Optional, List

from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent


def _is_docs_task(t: str) -> bool:
    t = t.lower()
    markers = [
        "readme", "installation", "install", "setup",
        "docs", "documentation", "doc", "guide",
        "getting started", "usage",
    ]
    ua = [
        "readme", "документац", "інсталяц", "встановлен",
        "гайд", "інструкц", "як запустити", "як користуватись",
    ]
    return any(m in t for m in markers) or any(m in t for m in ua)


def _is_install_section(t: str) -> bool:
    t = t.lower()
    return "installation" in t or "встанов" in t or "інсталяц" in t


def _is_readme_outline(t: str) -> bool:
    t = t.lower()
    return "readme" in t and ("outline" in t or "структур" in t or "мінімал" in t)


@register_agent
class DocsAgent(BaseAgent):
    """
    Спеціалізований агент для документації:
    README, installation, usage-приклади, короткі гайды.
    """

    name = "docs"
    description = "Допомагає з README, інсталяційними інструкціями та документацією."

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        t = task.lower()
        if _is_docs_task(t):
            return 0.98

        # якщо явно є слова про "написати інструкцію / гайд"
        markers = [
            "напиши інструкц", "напиши гайд", "зроби readme",
            "опиши встановлення", "опиши використання",
        ]
        if any(m in t for m in markers):
            return 0.9

        return 0.2  # не перехоплює все підряд

    def _readme_outline(self, task: str) -> str:
        lines: List[str] = [
            "## Мінімальний каркас README",
            "",
            "- Короткий опис проєкту (1–2 речення).",
            "- Розділ «Встановлення» (Installation).",
            "- Розділ «Використання» (Usage).",
            "- Налаштування / змінні оточення (якщо є).",
            "- Приклади команд.",
            "- Як запускати тести (якщо є).",
            "- Ліцензія та контакти / посилання.",
        ]
        return "\n".join(lines)

    def _install_section(self, task: str) -> str:
        lines: List[str] = [
            "## Встановлення",
            "",
            "### Попередні вимоги",
            "- Встановлений Python 3.10+.",
            "- Встановлений git.",
            "",
            "### Кроки",
            "1. Клонувати репозиторій:",
            "   ```bash",
            "   git clone <URL_РЕПОЗИТОРІЮ>",
            "   cd <КАТАЛОГ_ПРОЄКТУ>",
            "   ```",
            "2. Створити віртуальне середовище:",
            "   ```bash",
            "   python -m venv .venv",
            "   ```",
            "3. Активувати віртуальне середовище:",
            "   ```bash",
            "   source .venv/bin/activate  # Linux/macOS",
            "   .venv\\Scripts\\activate   # Windows",
            "   ```",
            "4. Встановити залежності:",
            "   ```bash",
            "   pip install -r requirements.txt",
            "   ```",
            "5. Перевірити запуск програми:",
            "   ```bash",
            "   python app.py --help",
            "   ```",
            "",
            "### Швидкий чекліст",
            "- [ ] Клонувати репозиторій і перейти в каталог проєкту.",
            "- [ ] Створити та активувати віртуальне середовище.",
            "- [ ] Встановити залежності.",
            "- [ ] Запустити `python app.py --help` та переконатися, що помилок немає.",
        ]
        return "\n".join(lines)

    def _generic_docs_help(self, task: str) -> str:
        lines: List[str] = [
            "## Документація / текст для проєкту",
            "",
            "### Що зробити",
            "- Коротко сформулюй, що це за проєкт і для чого він.",
            "- Опиши, як його встановити та запустити.",
            "- Додай 1–2 типових сценарії використання.",
            "- За потреби додай посилання на додаткові ресурси або тести.",
        ]
        return "\n".join(lines)

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        t = task.lower()

        if _is_readme_outline(t):
            out = self._readme_outline(task)
        elif _is_install_section(t):
            out = self._install_section(task)
        elif _is_docs_task(t):
            out = self._generic_docs_help(task)
        else:
            out = self._generic_docs_help(task)

        return AgentResult(
            agent=self.name,
            output=out,
            meta={"mode": "docs"},
        )