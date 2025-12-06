from typing import Optional
from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent


@register_agent
class WriterAgent(BaseAgent):
    name = "writer"
    description = "Helps with writing, rewriting, outlines, and clear structured text."

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        t = task.lower()
        keywords = [
            "write", "rewrite", "story", "outline", "essay",
            "text", "script", "email", "post", "article"
        ]
        hits = sum(1 for k in keywords if k in t)
        if hits == 0:
            return 0.1
        return min(1.0, 0.4 + hits * 0.15)

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        flags = memory.get("flags", {}) or {}
        force_structure = bool(flags.get("force_structure"))

        if force_structure:
            output = (
                "## Writing Task\n"
                f"{task}\n\n"
                "## Suggested Outline\n"
                "- Goal / audience\n"
                "- Key message\n"
                "- Structure of points\n"
                "- Tone and style\n\n"
                "## Draft Starter\n"
                "- Opening line idea\n"
                "- 3 main bullets to expand\n"
                "- Closing line idea"
            )
        else:
            output = (
                f"Writing task: {task}\n"
                "Outline: goal/audience -> key message -> 3 points -> closing.\n"
                "Draft starter: opening + 3 bullets + closing."
            )

        return AgentResult(agent=self.name, output=output, meta={"mode": "template"})
