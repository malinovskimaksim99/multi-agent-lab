from typing import Optional, Dict, Any, List
import re

from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent


def _has_heading(text: str) -> bool:
    return "## " in text or "\n#" in text


def _count_bullets(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip().startswith("- "))


def _looks_meta_template(text: str) -> bool:
    t = text.lower()
    patterns = [
        # analyst шаблон
        "identify the core request",
        "provide a concise answer",
        "include a quick self-check",

        # explainer шаблон
        "define each concept",
        "explain the purpose",
        "show a simple contrast",
        "summarize the takeaway",
        "concept a:",
        "concept b:",
        "key difference:",
        "when to choose each",

        # writer/meta
        "draft starter",
        "suggested outline",
        "approach:",
    ]
    return any(p in t for p in patterns)


def _looks_too_generic(text: str) -> bool:
    t = text.lower()
    generic_phrases = [
        "looks ok",
        "minor polish",
        "consider constraints",
        "kept concise",
        "merged from selected agents",
    ]
    return any(p in t for p in generic_phrases) and len(t) < 400


def _needs_steps(task: str) -> bool:
    t = task.lower()
    markers = [
        "plan", "planning", "strategy", "roadmap", "outline", "steps",
        "installation", "readme",
        "план", "сплануй", "крок", "дорожня карта", "встанов", "інсталяц"
    ]
    return any(m in t for m in markers)


@register_agent
class CriticAgent(BaseAgent):
    name = "critic"
    description = "Reviews drafts for quality issues and adds tags for learning/eval."

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        return 1.0

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        context = context or {}
        draft = context.get("draft", "") or ""

        notes: List[str] = []
        tags: List[str] = []

        if not draft.strip():
            notes.append("No draft provided for critique.")
            tags.append("missing_draft")
            return AgentResult(
                agent=self.name,
                output={"notes": notes, "tags": tags},
                meta={"ok": False, "reason": "empty_draft"}
            )

        # Structure
        if not _has_heading(draft) and _count_bullets(draft) < 2:
            notes.append("Add clearer structure (headings or bullet points).")
            tags.append("missing_structure")

        # Steps expectation
        if _needs_steps(task) and _count_bullets(draft) < 3:
            notes.append("Add more explicit steps/checklist items for this task.")
            tags.append("missing_steps")

        # Meta-template detection
        if _looks_meta_template(draft):
            notes.append("Response looks like a meta-template; add task-specific content.")
            tags.append("meta_template")

        # Too generic
        if _looks_too_generic(draft):
            notes.append("Answer may be too generic; add 2–3 concrete points.")
            tags.append("too_generic")

        # Length sanity
        if len(draft.strip()) < 120 and "summary" not in draft.lower():
            notes.append("Consider expanding slightly for clarity.")
            tags.append("too_short")

        if not notes:
            notes.append("Looks ok. Minor polish only.")

        return AgentResult(
            agent=self.name,
            output={"notes": notes, "tags": tags},
            meta={"ok": True, "tags_count": len(tags)}
        )
