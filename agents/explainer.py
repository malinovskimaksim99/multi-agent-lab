from typing import Optional, List
from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent


@register_agent
class ExplainerAgent(BaseAgent):
    name = "explainer"
    description = "Specialist for clear explanations, differences, and how/why questions."

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        t = task.lower()

        strong = [
            "explain", "difference", "compare", "why", "how",
            "what is", "roles of", "versus", "vs"
        ]
        if any(s in t for s in strong):
            return 0.92

        medium = [
            "meaning", "concept", "overview", "summary", "introduce"
        ]
        if any(m in t for m in medium):
            return 0.65

        return 0.15

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        flags = memory.get("flags", {}) or {}
        force_structure = bool(flags.get("force_structure"))
        expand_when_short = bool(flags.get("expand_when_short"))

        # Very lightweight “better-than-template” explanation scaffold
        key_points: List[str] = [
            "Define each concept in one sentence.",
            "Explain the purpose and when to use it.",
            "Show a simple contrast or example.",
            "Summarize the takeaway in 1–2 lines."
        ]

        if expand_when_short:
            key_points.append("Add one practical tip or common pitfall.")

        if force_structure:
            output = (
                "## Question\n"
                f"{task}\n\n"
                "## Explanation Map\n"
                + "\n".join(f"- {p}" for p in key_points) +
                "\n\n## Example (template)\n"
                "- Concept A: definition + purpose\n"
                "- Concept B: definition + purpose\n"
                "- Key difference: 1–2 bullets\n"
                "- When to choose each: 1–2 bullets\n"
            )
        else:
            output = (
                f"Question: {task}\n"
                "Approach: define terms -> purpose -> contrast -> quick example -> takeaway."
            )

        return AgentResult(agent=self.name, output=output, meta={"mode": "explain_scaffold"})
