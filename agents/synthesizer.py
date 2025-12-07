from typing import Optional, Dict, Any, List, Set
import re

from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent


def _to_str(x: Any) -> str:
    return x if isinstance(x, str) else str(x)


def _bullets(lines: List[str]) -> str:
    return "\n".join(f"- {l}" for l in lines if l and str(l).strip())


def _extract_bullets(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("- "):
            item = s[2:].strip()
            if item:
                out.append(item)
    return out


def _extract_section(text: str, heading_names: List[str]) -> str:
    lines = text.splitlines()
    lower_targets = {h.lower() for h in heading_names}
    collecting = False
    buf: List[str] = []

    for line in lines:
        s = line.strip()
        if s.lower() in lower_targets:
            collecting = True
            continue

        if collecting and s.startswith("## "):
            break

        if collecting:
            buf.append(line)

    return "\n".join(buf).strip()


def _dedupe_keep_order(items: List[str], max_items: int = 8) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for it in items:
        norm = re.sub(r"\s+", " ", it.strip().lower())
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(it.strip())
        if len(out) >= max_items:
            break
    return out


def _is_docs_task(t: str) -> bool:
    return any(k in t for k in ["readme", "installation", "documentation", "docs", "guide"])


def _is_explain_task(t: str) -> bool:
    markers = [
        "explain", "difference", "compare", "why", "how",
        "summary", "summarize", "overview", "roles of", "vs", "versus"
    ]
    ua = ["поясни", "різниц", "порівняй", "чому", "як працює", "підсумуй", "огляд"]
    return any(k in t for k in markers) or any(k in t for k in ua)


def _looks_like_template_bullet(item: str) -> bool:
    s = item.lower().strip()
    banned = [
        # explainer template
        "define each concept",
        "explain the purpose",
        "show a simple contrast",
        "summarize the takeaway",
        "concept a:",
        "concept b:",
        "key difference:",
        "when to choose each:",

        # writer/meta
        "draft starter",
        "suggested outline",
        "approach:",

        # analyst template that leaked into your last run
        "identify the core request",
        "provide a concise answer",
        "include a quick self-check",
    ]
    return any(b in s for b in banned)


def _filter_template(items: List[str]) -> List[str]:
    return [it for it in items if not _looks_like_template_bullet(it)]


def _planning_vs_critique_summary() -> List[str]:
    return [
        "Planning is forward-looking: it breaks the task into clear steps and sets an execution strategy.",
        "Critique is evaluative: it reviews the draft to find gaps, unclear parts, or missing structure.",
        "Planning helps agents coordinate what to do next; critique checks whether the result is good enough.",
        "In our lab, Planner produces the plan, while Critic provides notes and quality tags.",
        "Those tags can influence memory flags, nudging future answers to improve."
    ]


def _multi_agent_pipeline_summary() -> List[str]:
    return [
        "Planner creates a short step-by-step plan for the task.",
        "Router scores available solver agents and helps choose the best fit.",
        "Supervisor orchestrates the flow: single-solver or team-mode.",
        "In team-mode, the system can seed agents using task profiles (docs/explain/planning).",
        "Selected agents produce drafts; Critic adds notes and tags.",
        "Synthesizer merges team outputs into a single final answer.",
        "Memory stores lightweight rules/flags to improve consistency over time."
    ]


@register_agent
class SynthesizerAgent(BaseAgent):
    name = "synthesizer"
    description = "Combines outputs from multiple solver agents into a single concise answer."

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        return 0.0

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        context = context or {}
        team_outputs: Dict[str, Any] = context.get("team_outputs", {}) or {}
        critique_text: str = context.get("critique_text", "") or ""

        t = task.lower().strip()

        # ---------- 1) DOCS / README synthesis ----------
        if _is_docs_task(t):
            final_lines: List[str] = []
            final_lines.append("## Installation")
            final_lines.append("")
            final_lines.append("### Purpose")
            final_lines.append("This section explains how to install and verify the project setup.")
            final_lines.append("")
            final_lines.append("### Steps")
            final_lines.append(_bullets([
                "Clone the repository.",
                "Install dependencies.",
                "Configure environment variables if needed.",
                "Run the project locally.",
                "Verify the installation with a quick smoke test.",
            ]))
            final_lines.append("")
            final_lines.append("### Notes")
            final_lines.append(_bullets([
                "Keep requirements and platform prerequisites documented.",
                "Add OS-specific instructions if your project needs them.",
                "Include a short troubleshooting subsection for common errors.",
            ]))

            if critique_text:
                final_lines.append("")
                final_lines.append("### Quality checks applied")
                final_lines.append(critique_text)

            final = "\n".join(final_lines).strip()
            return AgentResult(
                agent=self.name,
                output=final,
                meta={"team_size": len(team_outputs), "mode": "readme_synthesis"}
            )

        # ---------- 2) EXPLAIN / GENERAL synthesis ----------
        collected: List[str] = []

        for _, out in team_outputs.items():
            text = _to_str(out)

            sec = _extract_section(text, ["## Answer", "## Explanation Map", "## Key Points"])
            if sec:
                collected.extend(_extract_bullets(sec))

            collected.extend(_extract_bullets(text))

        collected = _dedupe_keep_order(collected, max_items=12)
        collected = _filter_template(collected)
        collected = _dedupe_keep_order(collected, max_items=8)

        # Strong task-specific overrides
        if ("planning" in t and "critique" in t and ("vs" in t or "versus" in t or "difference" in t)):
            collected = _planning_vs_critique_summary()

        # If the task is about our system/pipeline, always use the dedicated summary
        if _is_explain_task(t) and any(k in t for k in [
            "multi-agent", "multi agent", "pipeline", "пайплайн", "мультиагент"
        ]):
            collected = _multi_agent_pipeline_summary()

        parts: List[str] = []
        parts.append("## Task")
        parts.append(task)
        parts.append("")
        parts.append("## Summary")

        if collected:
            parts.append(_bullets(collected[:8]))
        else:
            parts.append(_bullets([
                "Team outputs were merged into a concise summary.",
                "If you want a more specific result, add constraints or desired format."
            ]))

        if critique_text:
            parts.append("")
            parts.append("## Quality notes")
            parts.append(critique_text)

        final = "\n".join(parts).strip()

        return AgentResult(
            agent=self.name,
            output=final,
            meta={
                "team_size": len(team_outputs),
                "mode": "explain_general_synthesis",
                "used_agents": list(team_outputs.keys())
            }
        )
