from typing import Optional, List

from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent


def _is_docs(t: str) -> bool:
    return any(k in t for k in ["readme", "installation", "docs", "documentation", "guide"])


def _is_explain(t: str) -> bool:
    markers = ["explain", "difference", "compare", "roles of", "summary", "summarize", "overview", "vs", "versus"]
    ua = ["поясни", "різниц", "порівняй", "що таке", "огляд", "підсумуй"]
    return any(m in t for m in markers) or any(m in t for m in ua)


def _is_planning(t: str) -> bool:
    markers = ["plan", "planning", "roadmap", "outline", "next steps", "strategy"]
    ua = ["план", "кроки", "наступні кроки", "дорожня карта", "стратег"]
    return any(m in t for m in markers) or any(m in t for m in ua)


@register_agent
class AnalystAgent(BaseAgent):
    name = "analyst"
    description = "General problem solver focused on clarity, structure, and task-specific reasoning."

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        t = task.lower()
        if _is_docs(t):
            return 0.55  # writer should lead docs
        if _is_explain(t):
            return 0.70  # explainer often higher
        if _is_planning(t):
            return 0.85
        return 0.75

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        t = task.lower().strip()

        # Try to respect a lightweight structure flag
        flags = (memory.get("flags") or {})
        force_structure = bool(flags.get("force_structure", False))

        lines: List[str] = []

        if _is_docs(t):
            lines = [
                "Provide a short purpose line for this section.",
                "List the minimum install steps (clone, deps, env, run).",
                "Add one-line verification step.",
                "Optionally add OS-specific notes."
            ]

        elif "planning vs critique" in t or ("planning" in t and "critique" in t and ("vs" in t or "difference" in t)):
            lines = [
                "Planning defines what to do and in what order.",
                "Critique evaluates a draft and points out weaknesses.",
                "Planning is proactive; critique is corrective.",
                "Together they form a loop: plan → draft → critique → improve."
            ]

        elif "multi-agent" in t or "pipeline" in t:
            lines = [
                "Planner creates a plan for the task.",
                "Router ranks which agents fit best.",
                "Supervisor runs single or team mode.",
                "Critic adds notes/tags.",
                "Synthesizer merges team outputs into a final answer.",
                "Memory stores lightweight flags to improve future runs."
            ]

        elif _is_planning(t):
            lines = [
                "Clarify the goal and success criteria.",
                "List 3–5 concrete steps.",
                "Add a quick test/verification step.",
                "Note likely risks or common failure points."
            ]

        elif _is_explain(t):
            lines = [
                "Define the key terms briefly.",
                "Highlight the most important difference/purpose.",
                "Give a small example.",
                "End with a one-line takeaway."
            ]

        else:
            # Generic but not meta-template phrases
            lines = [
                "Restate the task in your own words to confirm scope.",
                "Provide 3–5 concrete points that directly answer it.",
                "Add a short example or suggestion if relevant."
            ]

        if force_structure:
            out = (
                "## Task\n"
                f"{task}\n\n"
                "## Key points\n"
                + "\n".join(f"- {l}" for l in lines)
            )
        else:
            out = "\n".join(f"- {l}" for l in lines)

        return AgentResult(
            agent=self.name,
            output=out,
            meta={"mode": "content_scaffold"}
        )
