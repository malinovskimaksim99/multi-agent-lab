# multi-agent-lab
MVP multi-agent system with memory and self-critique

This server now supports two chat modes: `head` (default) and `writer`. Use `POST /chat` with a JSON body like `{"task":"...", "mode":"writer"}` to route the request to the Writer LLM, which uses per-project `llm.writer_model` from the database (and the same base URL as other LM Studio calls).

In the built-in web UI, there is a mode switch next to the `auto` checkbox. Choose **Writer** to send the message through the writer model; the response will be labeled “WriterAgent”.

When you call writer mode, the HeadAgent records a short shadow note in `head_notes` (what was requested, a brief summary of the writer’s reply, and whether follow-up actions may be needed). This keeps a lightweight audit trail without changing the main chat flow.
