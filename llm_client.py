import json
import os
import socket
from urllib import error, request


def env_default_base_url() -> str:
    return os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")


def _truncate_text(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "... (truncated)"


def chat_openai_compat(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int | None = None,
    timeout_s: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            body_bytes = resp.read()
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP error {exc.code}: {_truncate_text(body)}"
        ) from exc
    except (error.URLError, TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc

    body_text = body_bytes.decode("utf-8", errors="replace")
    try:
        data = json.loads(body_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON response: {_truncate_text(body_text)}"
        ) from exc

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content")
    )
    if not content:
        raise ValueError(
            f"Empty response content: {_truncate_text(body_text)}"
        )

    return content
