"""
Minimal HTTP API wrapper around multi-agent-lab.

Запускається так (з кореня репозиторію):
    uvicorn server:app --reload

Цей сервер просто викликає `app.py` як CLI,
щоб не лізти у внутрішню реалізацію Supervisor / HeadAgent.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import subprocess
import sys
from typing import Optional

from db import get_projects, get_book_outline


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
            .projects-list {
                margin-top: 8px;
                display: flex;
                flex-direction: column;
                gap: 4px;
            }
            .project-item {
                border-radius: 6px;
                padding: 6px 8px;
                background: #020617;
                border: 1px solid #1f2933;
                cursor: default;
            }
            .project-item.active {
                border-color: #22c55e;
                box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.3);
            }
            .project-name {
                font-size: 13px;
                color: #e5e7eb;
            }
            .project-meta {
                font-size: 11px;
                color: #9ca3af;
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
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
            }
            .header-text {
                flex: 1;
            }
            .structure-panel {
                border-bottom: 1px solid #1f2933;
                background: #020617;
                padding: 8px 16px 12px;
                max-height: 260px;
                overflow-y: auto;
                box-sizing: border-box;
                font-size: 13px;
            }
            .structure-panel.hidden {
                display: none;
            }
            .structure-header {
                font-size: 13px;
                color: #9ca3af;
                margin-bottom: 4px;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }
            .structure-header-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 4px;
            }
            .btn-ghost {
                border-radius: 6px;
                border: 1px solid #374151;
                padding: 2px 8px;
                font-size: 11px;
                cursor: pointer;
                background: transparent;
                color: #9ca3af;
            }
            .btn-ghost:hover {
                border-color: #4b5563;
                color: #e5e7eb;
            }
            .structure-body {
                color: #e5e7eb;
            }
            .btn-secondary {
                border-radius: 8px;
                border: 1px solid #374151;
                padding: 6px 10px;
                font-size: 12px;
                cursor: pointer;
                background: #020617;
                color: #e5e7eb;
            }
            .btn-secondary:hover {
                border-color: #4b5563;
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
                <strong>Проєкти:</strong>
                <div id="projects-list" class="projects-list">
                    <!-- Тимчасово — один проєкт. Після fetch /projects список оновиться. -->
                    <div class="project-item active">
                        <div class="project-name">Розробка</div>
                        <div class="project-meta">dev</div>
                    </div>
                </div>
                <small>Поки що цей UI працює з поточним проєктом.</small>
            </div>
        </div>
        <div class="main">
            <div class="header">
                <div class="header-text">
                    Простий чат з /chat (app.py). Далі будемо розвивати до повного робочого середовища.
                </div>
                <button id="toggle-structure" class="btn-secondary">Структура ▼</button>
            </div>
            <div id="structure-panel" class="structure-panel hidden">
                <div class="structure-header-row">
                    <div class="structure-header">Структура проєкту</div>
                    <button id="refresh-outline" class="btn-ghost">Оновити</button>
                </div>
                <div class="structure-body">
                    <p>Тут буде дерево: Книга → Глави → Сцени та ключові сюжетні розгалуження.</p>
                    <p>Якщо структура не оновилась, натисни «Оновити».</p>
                </div>
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
            // Тимчасово фіксований id тестової книги для панелі структури.
            // Пізніше зробимо вибір книги за поточним проєктом.
            const OUTLINE_BOOK_ID = 2;

            const taskEl = document.getElementById('task');
            const autoEl = document.getElementById('auto');
            const sendBtn = document.getElementById('send');
            const logEl = document.getElementById('log');
            const statusEl = document.getElementById('status');
            const toggleStructureBtn = document.getElementById('toggle-structure');
            const structurePanel = document.getElementById('structure-panel');
            const structureBody = document.querySelector('.structure-body');
            const refreshOutlineBtn = document.getElementById('refresh-outline');
            const projectsListEl = document.getElementById('projects-list');
            let structureVisible = false;
            let outlineLoaded = false;
            // Поточний проєкт, обраний у сайдбарі
            let currentProjectId = null;
            let currentProjectType = null;

            async function loadProjects() {
                if (!projectsListEl) return;

                try {
                    const resp = await fetch('/projects');
                    if (!resp.ok) {
                        throw new Error('HTTP ' + resp.status);
                    }
                    const data = await resp.json();
                    renderProjects(data.projects || []);
                } catch (err) {
                    console.error(err);
                    projectsListEl.innerHTML =
                        '<div class="project-item"><div class="project-name">Помилка завантаження проєктів</div></div>';
                }
            }

            function renderProjects(projects) {
                if (!projectsListEl) return;

                if (!projects.length) {
                    projectsListEl.innerHTML =
                        '<div class="project-item"><div class="project-name">Немає проєктів</div></div>';
                    currentProjectId = null;
                    currentProjectType = null;
                    return;
                }

                let html = '';
                for (const p of projects) {
                    const isActive = (p.status === 'active');
                    // Якщо ще не обрано поточний проєкт — беремо перший active
                    if (isActive && currentProjectId === null) {
                        currentProjectId = p.id;
                        currentProjectType = p.type || null;
                    }
                    html += '<div class="project-item' + (isActive ? ' active' : '') + '" ' +
                            'data-project-id="' + p.id + '" ' +
                            'data-project-type="' + (p.type || '') + '">';
                    html += '<div class="project-name">' + (p.name || 'Без назви') + '</div>';
                    if (p.type) {
                        html += '<div class="project-meta">' + p.type + '</div>';
                    }
                    html += '</div>';
                }
                projectsListEl.innerHTML = html;

                // Навішуємо клік‑обробники для вибору поточного проєкту
                const items = projectsListEl.querySelectorAll('.project-item');
                items.forEach((el) => {
                    el.addEventListener('click', () => {
                        const pid = el.getAttribute('data-project-id');
                        const ptype = el.getAttribute('data-project-type') || null;
                        currentProjectId = pid ? parseInt(pid, 10) : null;
                        currentProjectType = ptype;

                        items.forEach((i) => i.classList.remove('active'));
                        el.classList.add('active');

                        // Якщо відкрита панель структури — оновлюємо її для вибраного проєкту
                        if (structureVisible) {
                            loadOutlineForCurrentProject();
                        }
                    });
                });
            }

            if (toggleStructureBtn && structurePanel) {
                toggleStructureBtn.addEventListener('click', () => {
                    structureVisible = !structureVisible;
                    if (structureVisible) {
                        structurePanel.classList.remove('hidden');
                        toggleStructureBtn.textContent = 'Структура ▲';
                        // Перший раз при відкритті — завантажуємо структуру
                        if (!outlineLoaded) {
                            loadOutlineForCurrentProject();
                        }
                    } else {
                        structurePanel.classList.add('hidden');
                        toggleStructureBtn.textContent = 'Структура ▼';
                    }
                });
            }

            if (refreshOutlineBtn && structurePanel) {
                refreshOutlineBtn.addEventListener('click', () => {
                    loadOutlineForCurrentProject();
                });
            }
            function loadOutlineForCurrentProject() {
                if (!structureBody) return;

                // Поки що для non-writing проєктів показуємо простий текст.
                if (!currentProjectId) {
                    structureBody.innerHTML = '<p>Поточний проєкт не вибрано.</p>';
                    return;
                }
                if (currentProjectType !== 'writing') {
                    structureBody.innerHTML =
                        '<p>Для проєкту типу <code>' + (currentProjectType || 'unknown') +
                        '</code> структура книги ще не налаштована.</p>';
                    return;
                }

                // Тимчасово: використовуємо фіксований OUTLINE_BOOK_ID.
                // Пізніше підʼєднаємо справжній пошук книги за проєктом.
                loadOutline(OUTLINE_BOOK_ID);
            }

            async function loadOutline(bookId) {
                if (!structureBody) return;
                structureBody.innerHTML = '<p>Завантаження структури…</p>';

                try {
                    const resp = await fetch('/writing/outline?book_id=' + bookId);
                    if (!resp.ok) {
                        throw new Error('HTTP ' + resp.status);
                    }
                    const data = await resp.json();
                    renderOutline(data);
                    outlineLoaded = true;
                } catch (err) {
                    console.error(err);
                    structureBody.innerHTML =
                        '<p>Не вдалося завантажити структуру: ' + err.message + '</p>';
                }
            }

            function renderOutline(data) {
                if (!structureBody) return;

                if (!data || !data.book) {
                    structureBody.innerHTML = '<p>Структура відсутня.</p>';
                    return;
                }

                const book = data.book;
                const chapters = data.chapters || [];
                let html = '';

                html += '<div><strong>Книга:</strong> ' + book.title +
                        ' <span style="color:#9ca3af;">[' + (book.status || 'unknown') + ']</span></div>';

                if (book.project_name) {
                    html += '<div style="font-size:12px;color:#9ca3af;">Проєкт: ' +
                            book.project_name + '</div>';
                }

                if (book.synopsis) {
                    html += '<p style="margin-top:4px;">' + book.synopsis + '</p>';
                }

                if (!chapters.length) {
                    html += '<p>Глави ще не додані.</p>';
                } else {
                    html += '<ul style="margin:8px 0 0 0; padding-left:16px;">';
                    for (const ch of chapters) {
                        html += '<li>';
                        html += '<div><strong>Глава ' + (ch.number || '') + ':</strong> ' +
                                (ch.title || '') +
                                ' <span style="color:#9ca3af;">[' + (ch.status || 'unknown') + ']</span></div>';

                        if (ch.summary) {
                            html += '<div style="font-size:12px;color:#9ca3af;margin-bottom:2px;">' +
                                    ch.summary + '</div>';
                        }

                        const scenes = ch.scenes || [];
                        if (scenes.length) {
                            html += '<ul style="margin:4px 0 4px 16px;padding-left:12px;">';
                            for (const sc of scenes) {
                                html += '<li>';
                                html += '<span>' + (sc.title || 'Сцена') +
                                        ' <span style="color:#9ca3af;">[' +
                                        (sc.status || 'unknown') + ']</span></span>';
                                html += '</li>';
                            }
                            html += '</ul>';
                        }

                        html += '</li>';
                    }
                    html += '</ul>';
                }

                structureBody.innerHTML = html;
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

            // При завантаженні сторінки одразу підтягуємо список проєктів
            loadProjects();
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


# New endpoints for listing projects and getting book outline
@app.get("/projects")
async def list_projects() -> dict:
    """
    Повертає список усіх проєктів для UI.
    """
    projects = get_projects()
    return {"projects": projects}


@app.get("/writing/outline")
async def writing_outline(book_id: int):
    """
    Повертає структуру книги (outline) за book_id.
    Використовується UI для відображення книги: заголовок, глави, сцени.
    """
    try:
        outline = get_book_outline(book_id)
    except ValueError as exc:
        # Якщо книга не знайдена, повертаємо 404
        raise HTTPException(status_code=404, detail=str(exc))
    return outline


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)