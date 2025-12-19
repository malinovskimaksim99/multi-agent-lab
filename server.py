

"""
Minimal HTTP API wrapper around multi-agent-lab.

Запускається так (з кореня репозиторію):
    uvicorn server:app --reload

Цей сервер просто викликає `app.py` як CLI,
щоб не лізти у внутрішню реалізацію Supervisor / HeadAgent.
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import subprocess
import sys
from typing import Optional


app = FastAPI(title="multi-agent-lab API")


class ChatRequest(BaseModel):
    """Запит на запуск одного завдання multi-agent-lab."""

    task: str
    auto: bool = True


class ChatResponse(BaseModel):
    """Відповідь із результатом запуску app.py."""

    task: str
    auto: bool
    stdout: str
    stderr: str
    return_code: int


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """
    Проста HTML-сторінка з мінімальним чат-інтерфейсом до /chat.

    Це тимчасовий "shell UI", щоб можна було клікати,
    не лізучи щоразу в /docs або curl.
    """
    return """
    <!doctype html>
    <html lang="uk">
    <head>
        <meta charset="utf-8" />
        <title>multi-agent-lab — Chat</title>
        <style>
            body {
                margin: 0;
                padding: 0;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                display: flex;
                height: 100vh;
            }
            .sidebar {
                width: 220px;
                background: #111827;
                color: #e5e7eb;
                padding: 16px;
                box-sizing: border-box;
            }
            .sidebar h1 {
                font-size: 16px;
                margin: 0 0 12px;
            }
            .sidebar small {
                display: block;
                color: #9ca3af;
                margin-top: 4px;
            }
            .main {
                flex: 1;
                display: flex;
                flex-direction: column;
                background: #020617;
                color: #e5e7eb;
            }
            .header {
                padding: 12px 16px;
                border-bottom: 1px solid #1f2933;
                font-size: 14px;
                color: #9ca3af;
            }
            .chat-log {
                flex: 1;
                padding: 16px;
                overflow-y: auto;
                box-sizing: border-box;
            }
            .msg {
                margin-bottom: 12px;
            }
            .msg.me {
                text-align: right;
            }
            .msg .who {
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: #9ca3af;
            }
            .msg .bubble {
                display: inline-block;
                border-radius: 12px;
                padding: 8px 10px;
                margin-top: 4px;
                max-width: 80%;
                text-align: left;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .msg.me .bubble {
                background: #1d4ed8;
                color: white;
            }
            .msg.bot .bubble {
                background: #020617;
                border: 1px solid #1f2933;
            }
            .input-bar {
                border-top: 1px solid #1f2933;
                padding: 10px 12px;
                display: flex;
                flex-direction: column;
                gap: 8px;
                box-sizing: border-box;
            }
            .row {
                display: flex;
                gap: 8px;
                align-items: center;
            }
            textarea {
                flex: 1;
                resize: none;
                min-height: 60px;
                max-height: 160px;
                border-radius: 8px;
                border: 1px solid #374151;
                background: #020617;
                color: #e5e7eb;
                padding: 8px 10px;
                font-family: inherit;
                font-size: 14px;
            }
            button {
                border-radius: 8px;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                cursor: pointer;
                background: #22c55e;
                color: #022c22;
                font-weight: 500;
            }
            button:disabled {
                opacity: 0.6;
                cursor: default;
            }
            .meta-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 12px;
                color: #9ca3af;
            }
            .status {
                font-size: 12px;
            }
            .status.error {
                color: #f97373;
            }
            input[type="checkbox"] {
                margin-right: 4px;
            }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h1>multi-agent-lab</h1>
            <div>
                <strong>Проєкт:</strong>
                <div id="current-project">Розробка</div>
                <small>Поки що цей UI працює з поточним проєктом.</small>
            </div>
        </div>
        <div class="main">
            <div class="header">
                Простий чат з /chat (app.py). Далі будемо розвивати до повного робочого середовища.
            </div>
            <div id="log" class="chat-log"></div>
            <div class="input-bar">
                <div class="row">
                    <textarea id="task" placeholder="Напишіть запит, наприклад: Склади план розвитку multi-agent-lab."></textarea>
                    <button id="send">Send</button>
                </div>
                <div class="meta-row">
                    <label>
                        <input type="checkbox" id="auto" checked />
                        auto (--auto)
                    </label>
                    <div id="status" class="status"></div>
                </div>
            </div>
        </div>
        <script>
            const taskEl = document.getElementById('task');
            const autoEl = document.getElementById('auto');
            const sendBtn = document.getElementById('send');
            const logEl = document.getElementById('log');
            const statusEl = document.getElementById('status');

            function appendMessage(who, text) {
                const div = document.createElement('div');
                div.className = 'msg ' + (who === 'You' ? 'me' : 'bot');
                const whoEl = document.createElement('div');
                whoEl.className = 'who';
                whoEl.textContent = who;
                const bubble = document.createElement('div');
                bubble.className = 'bubble';
                bubble.textContent = text;
                div.appendChild(whoEl);
                div.appendChild(bubble);
                logEl.appendChild(div);
                logEl.scrollTop = logEl.scrollHeight;
            }

            async function send() {
                const task = taskEl.value.trim();
                if (!task) return;
                sendBtn.disabled = true;
                statusEl.textContent = 'Виконується...';
                statusEl.classList.remove('error');

                appendMessage('You', task);
                taskEl.value = '';

                try {
                    const resp = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            task: task,
                            auto: autoEl.checked
                        })
                    });
                    if (!resp.ok) {
                        throw new Error('HTTP ' + resp.status);
                    }
                    const data = await resp.json();
                    const text = data.stdout || '(порожній stdout)';
                    appendMessage('lab', text);
                    statusEl.textContent = 'Готово (код ' + data.return_code + ')';
                } catch (err) {
                    console.error(err);
                    statusEl.textContent = 'Помилка: ' + err.message;
                    statusEl.classList.add('error');
                    appendMessage('lab', 'Помилка при виклику /chat: ' + err.message);
                } finally {
                    sendBtn.disabled = false;
                    taskEl.focus();
                }
            }

            sendBtn.addEventListener('click', send);
            taskEl.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault();
                    send();
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health() -> dict:
    """Простий health-check ендпоінт."""

    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Запустити одне завдання через `app.py` як через CLI.

    Ми спеціально не імпортуємо Supervisor / HeadAgent,
    а просто викликаємо існуючий інтерфейс командного рядка:
        python app.py --task "..." --auto
    """

    # Формуємо команду для підпроцесу
    cmd = [sys.executable, "app.py", "--task", req.task]
    if req.auto:
        cmd.append("--auto")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    return ChatResponse(
        task=req.task,
        auto=req.auto,
        stdout=proc.stdout,
        stderr=proc.stderr,
        return_code=proc.returncode,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)