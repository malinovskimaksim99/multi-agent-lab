from typing import List, Dict, Any, Optional
from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent


class Analyst:
    def draft(self, task: str, plan: List[str], memory: Dict[str, Any]) -> str:
        flags = memory.get("flags", {}) or {}
        force_structure = bool(flags.get("force_structure"))
        expand_when_short = bool(flags.get("expand_when_short"))

        base_points = [
            "Identify the core request and constraints.",
            "Provide a concise answer with actionable points.",
            "Include a quick self-check for gaps."
        ]

        if expand_when_short:
            base_points.append("Add one short example or mini-checklist.")
            base_points.append("State assumptions and limits of the answer.")

        if force_structure:
            body = (
                "## Task\n"
                f"{task}\n\n"
                "## Plan\n"
                + "\n".join(f"{i+1}. {p}" for i, p in enumerate(plan))
                + "\n\n## Answer\n"
                + "\n".join(f"- {p}" for p in base_points)
            )
        else:
            body = (
                f"Task: {task}\n"
                "Plan: " + ", ".join(plan) + "\n"
                "Answer: " + " ".join(base_points)
            )

        return body

    def revise(self, draft: str, critique_text: str) -> str:
        if "##" in draft:
            return draft + "\n\n## Revisions\n" + critique_text
        return draft + "\n\nRevisions:\n" + critique_text


@register_agent
class AnalystAgent(BaseAgent):
    name = "analyst"
    description = "Produces a draft answer using plan and memory flags."

    def __init__(self):
        self._analyst = Analyst()

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        t = task.lower()

        writing_markers = [
            "write", "rewrite", "story", "outline", "essay",
            "readme", "documentation", "doc", "guide", "installation"
        ]
        if any(m in t for m in writing_markers):
            return 0.55

        analysis_markers = [
            "analyze", "analysis", "compare", "pros", "cons",
            "explain", "evaluate", "reason", "checklist", "plan"
        ]
        hits = sum(1 for m in analysis_markers if m in t)

        return min(0.9, 0.6 + hits * 0.07)

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        plan: List[str] = []
        if context and isinstance(context.get("plan"), list):
            plan = context["plan"]

        draft = self._analyst.draft(task, plan, memory)
        return AgentResult(agent=self.name, output=draft, meta={})

    def revise(self, draft: str, critique_text: str) -> str:
        return self._analyst.revise(draft, critique_text)
