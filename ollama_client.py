# ollama_client.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib import request, error

# Базова адреса локального сервера Ollama
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    """Помилка при виклику локальної Ollama-моделі."""
    pass


def chat_ollama(
    model: str,
    messages: List[Dict[str, str]],
    *,
    stream: bool = False,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Проста обгортка над Ollama /api/chat.

    - model    — назва моделі, напр. "qwen3-vl:8b"
    - messages — список {"role": "system"|"user"|"assistant", "content": "..."}
    - stream   — для простоти тримаємо False (отримуємо одну відповідь)
    - temperature, max_tokens — тонке налаштування генерації
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
        },
    }
    if max_tokens is not None:
        # Для Ollama це поле називається num_predict
        payload["options"]["num_predict"] = max_tokens

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=600) as resp:
            body = resp.read()
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
    if not isinstance(content, str):
        raise OllamaError(f"Немає тексту відповіді від Ollama: {obj}")
    return content


def call_head_llm(
    user_text: str,
    system_prompt: str,
    *,
    model: str = "qwen3-vl:8b",
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
    )