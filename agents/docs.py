from typing import Optional, List, Dict, Any

from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent
from db import get_agent_config


def _load_agent_config(agent_name: str) -> Dict[str, Any]:
    """
    Завантажує конфіг агента з БД через get_agent_config.
    Повертає словник з ключами типу 'style', 'max_lines' тощо.
    """
    cfg: Dict[str, Any] = {}
    try:
        style = get_agent_config(agent_name, "style")
        if style:
            cfg["style"] = style
    except Exception:
        pass

    try:
        max_lines = get_agent_config(agent_name, "max_lines")
        if max_lines not in (None, ""):
            try:
                cfg["max_lines"] = int(max_lines)
            except Exception:
                pass
    except Exception:
        pass

    return cfg


def _apply_style_to_text(text: str, config: Dict[str, Any]) -> str:
    """
    Застосовує стиль форматування до готового тексту документації:
      - style="compact" прибирає зайві порожні рядки;
      - max_lines скорочує відповідь до N рядків, якщо потрібно.
    """
    if not text:
        return text

    style = config.get("style")
    max_lines = config.get("max_lines")

    lines = text.splitlines()

    # Обрізаємо по кількості рядків, якщо задано
    if isinstance(max_lines, int) and max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]

    # Компактний стиль: прибираємо дублікати порожніх рядків
    if style == "compact":
        cleaned: List[str] = []
        prev_blank = False
        for ln in lines:
            s = ln.rstrip()
            is_blank = (s == "")
            if is_blank and prev_blank:
                continue
            cleaned.append(s)
            prev_blank = is_blank
        lines = cleaned

    return "\n".join(lines)


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
    description = (
        "Спеціаліст по README, розділах 'Встановлення', 'Використання' "
        "та іншій технічній документації."
    )

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        """
        Оцінює, наскільки задача схожа на документацію / README.
        Використовує:
          - маркери в тексті (англ + укр),
          - можливий task_type з контексту.
        """
        t = task.lower()
        ctx: Context = context or {}
        task_type = ctx.get("task_type")

        doc_markers_en = [
            "readme", "documentation", "docs", "guide",
            "install", "installation", "setup",
            "usage", "how to run", "quick start",
        ]
        doc_markers_ua = [
            "документац", "гайд", "інструкц", "опис",
            "встановлен", "інсталяц", "налаштуван",
            "запуск", "як запустити", "приклади команд",
        ]

        # Базовий score за ключові слова
        if any(m in t for m in doc_markers_en) or any(m in t for m in doc_markers_ua):
            # Для документації docs має бути вище writer'а
            score = 1.30
        else:
            # Якщо явно є слова про "написати інструкцію / гайд"
            markers = [
                "напиши інструкц", "напиши гайд", "зроби readme",
                "опиши встановлення", "опиши використання",
            ]
            if any(m in t for m in markers):
                score = 1.10
            else:
                # Майже не чіпаємо не-документаційні задачі
                score = 0.20

        # Підлаштування під task_type з Supervisor/Router
        if task_type == "docs":
            # Якщо вже класифіковано як docs — робимо його ще сильнішим.
            score = max(score, 1.50)
        elif task_type == "explain":
            # Якщо це пояснення, але все ж про документацію — не конкуруємо агресивно з explainer.
            score = max(score, 0.60)

        return score

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
            "   .venв\\Scripts\\activate   # Windows",
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
        t = task.lower().strip()

        flags = memory.get("flags", {}) or {}
        force_structure = bool(flags.get("force_structure"))
        # expand_when_short можна використати пізніше для додаткових секцій
        expand_when_short = bool(flags.get("expand_when_short"))

        # Завантажуємо конфіг для docs-агента з БД
        config = _load_agent_config(self.name)

        if _is_readme_outline(t):
            output = self._readme_outline(task)
            mode = "docs_readme_outline"
        elif _is_install_section(t):
            output = self._install_section(task)
            mode = "docs_install"
        elif _is_docs_task(t):
            output = self._generic_docs_help(task)
            mode = "docs_generic"
        else:
            # Фолбек: теж generic, щоб не повертати порожню відповідь
            output = self._generic_docs_help(task)
            mode = "docs_generic"

        styled_output = _apply_style_to_text(output, config)
        return AgentResult(
            agent=self.name,
            output=styled_output,
            meta={"mode": mode},
        )