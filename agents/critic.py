from typing import Dict, Any, List, Optional
from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent


class Critic:
    def _has_structure(self, text: str) -> bool:
        return ("##" in text or "\n-" in text or "\n1." in text or "\n2." in text)

    def review_structured(self, draft: str) -> Dict[str, Any]:
        notes: List[str] = []
        tags: List[str] = []

        if not self._has_structure(draft):
            notes.append("Add clearer structure (headings/bullets).")
            tags.append("structure")

        if len(draft.strip()) < 200:
            notes.append("Too short; add a bit more concrete steps.")
            tags.append("too_short")

        if not notes:
            notes.append("Looks ok. Minor polish only.")

        return {"notes": notes, "tags": tags}

    def review(self, draft: str) -> str:
        data = self.review_structured(draft)
        return "\n".join(f"- {n}" for n in data["notes"])


@register_agent
class CriticAgent(BaseAgent):
    name = "critic"
    description = "Reviews a draft answer and returns notes + quality tags."

    def __init__(self):
        self._critic = Critic()

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        return 0.7

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        draft = ""
        if context and isinstance(context.get("draft"), str):
            draft = context["draft"]

        data = self._critic.review_structured(draft)
        return AgentResult(agent=self.name, output=data, meta={})
