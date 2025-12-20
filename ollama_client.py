# ollama_client.py
from __future__ import annotations

import json
import os
import socket
from typing import Any, Dict, List, Optional
from urllib import request, error

# Базова адреса локального сервера Ollama
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Таймаут запиту до Ollama.
# Якщо OLLAMA_TIMEOUT_S НЕ задано (або задано як 0), запит може чекати без обмеження.
_raw_timeout = os.getenv("OLLAMA_TIMEOUT_S")
try:
    DEFAULT_TIMEOUT_S: Optional[float] = None if (_raw_timeout is None or _raw_timeout.strip() in ("", "0")) else float(_raw_timeout)
except Exception:
    DEFAULT_TIMEOUT_S = None


class OllamaError(RuntimeError):
    """Помилка при виклику локальної Ollama-моделі."""
    pass


def _sanitize_ollama_obj(obj: Any) -> Any:
    """Прибирає потенційно великі/службові поля (наприклад, reasoning/thinking) з об'єкта відповіді.

    Важливо: ми не хочемо друкувати thinking у повідомленнях про помилку.
    """
    try:
        if isinstance(obj, dict):
            obj2 = dict(obj)
            msg = obj2.get("message")
            if isinstance(msg, dict):
                msg2 = dict(msg)
                if "thinking" in msg2:
                    msg2["thinking"] = "<omitted>"
                obj2["message"] = msg2
            return obj2
    except Exception:
        pass
    return obj


def chat_ollama(
    model: str,
    messages: List[Dict[str, str]],
    *,
    stream: bool = False,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    num_ctx: Optional[int] = None,
    timeout_s: Optional[float] = None,
    _allow_retry: bool = True,
) -> str:
    """
    Проста обгортка над Ollama /api/chat.

    - model    — назва моделі, напр. "qwen3:8b"
    - messages — список {"role": "system"|"user"|"assistant", "content": "..."}
    - stream   — для простоти тримаємо False (отримуємо одну відповідь)
    - temperature, max_tokens — тонке налаштування генерації
    """
    options: Dict[str, Any] = {
        "temperature": temperature,
    }

    # Не нав'язуємо ліміти: додаємо лише якщо явно передали.
    if num_ctx is not None:
        options["num_ctx"] = int(num_ctx)
    if max_tokens is not None:
        # Для Ollama це поле називається num_predict
        options["num_predict"] = int(max_tokens)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": options,
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        effective_timeout = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S
        if effective_timeout is None:
            with request.urlopen(req) as resp:
                body = resp.read()
        else:
            with request.urlopen(req, timeout=float(effective_timeout)) as resp:
                body = resp.read()
    except (socket.timeout, TimeoutError) as e:
        raise OllamaError(
            "Ollama не відповіла вчасно (timeout). "
            "Спробуй зменшити промпт/модель або збільшити OLLAMA_TIMEOUT_S."
        ) from e
    except error.URLError as e:
        raise OllamaError(f"Не вдалося підключитись до Ollama: {e}") from e

    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        preview = body[:200]
        raise OllamaError(
            f"Неправильна відповідь від Ollama: {e}; body={preview!r}"
        ) from e

    if "error" in obj:
        raise OllamaError(f"Ollama повернула помилку: {obj['error']}")

    msg = obj.get("message", {})
    content = msg.get("content")

    # Нормальний шлях для /api/chat
    if isinstance(content, str) and content.strip():
        return content

    # Фолбек: інколи відповідь приходить у полі "response" (схоже на /api/generate)
    alt = obj.get("response")
    if isinstance(alt, str) and alt.strip():
        return alt

    # Деякі моделі/збірки Ollama можуть повертати reasoning у `message.thinking`,
    # але при цьому лишати `message.content` порожнім. Нам потрібен фінальний текст,
    # тому робимо ОДНУ спробу перезапиту з чіткою інструкцією "дай відповідь в content".
    thinking = msg.get("thinking") if isinstance(msg, dict) else None
    if _allow_retry and isinstance(thinking, str) and thinking.strip():
        retry_messages = list(messages) + [
            {
                "role": "user",
                "content": "Відповідай лише фінальним текстом (без thinking). Дай коротку відповідь у content.",
            }
        ]
        return chat_ollama(
            model=model,
            messages=retry_messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            timeout_s=timeout_s,
            _allow_retry=False,
        )

    # Якщо content порожній — вважаємо це помилкою, але не показуємо thinking у прев'ю.
    safe_obj = _sanitize_ollama_obj(obj)
    preview = json.dumps(safe_obj, ensure_ascii=False)[:500]
    raise OllamaError(f"Порожня або відсутня відповідь від Ollama: {preview}")


def call_head_llm(
    user_text: str,
    system_prompt: str,
    *,
    model: str = "qwen3:8b",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Зручна функція спеціально для HeadAgent:
    будує messages = [system, user] і викликає chat_ollama.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return chat_ollama(
        model=model,
        messages=messages,
        stream=False,
        temperature=temperature,
        max_tokens=max_tokens,
        # Без дефолтних лімітів: контекст/довжину контролює сама модель або явні параметри.
        num_ctx=None,
        timeout_s=None,
    )