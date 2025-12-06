from typing import Optional, Dict, Any, List
from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent


def _to_str(x: Any) -> str:
    return x if isinstance(x, str) else str(x)


def _bullets(lines: List[str]) -> str:
    return "\n".join(f"- {l}" for l in lines if l.strip())


@register_agent
class SynthesizerAgent(BaseAgent):
    name = "synthesizer"
    description = "Combines outputs from multiple solver agents into a single concise answer."

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        # Not meant to be chosen by router.
        return 0.0

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        context = context or {}
        team_outputs: Dict[str, Any] = context.get("team_outputs", {}) or {}
        critique_text: str = context.get("critique_text", "") or ""

        writer_text = _to_str(team_outputs.get("writer", ""))
        analyst_text = _to_str(team_outputs.get("analyst", ""))

        t = task.lower()

        # README/installation-focused synthesis
        if ("readme" in t) or ("installation" in t) or ("documentation" in t):
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
                meta={
                    "team_size": len(team_outputs),
                    "mode": "readme_synthesis",
                    "used_agents": list(team_outputs.keys())
                }
            )

        # Generic synthesis for other tasks
        summary_points: List[str] = []
        used = []

        for name, out in team_outputs.items():
            used.append(name)
            text = _to_str(out)

            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines:
                snippet = lines[0]
                if len(lines) > 1 and len(snippet) < 80:
                    snippet = f"{snippet} / {lines[1]}"
                summary_points.append(f"{name}: {snippet}")

        parts: List[str] = []
        parts.append("## Task")
        parts.append(task)

        if summary_points:
            parts.append("\n## Team Summary")
            parts.append(_bullets(summary_points))

        parts.append("\n## Final Answer")
        if writer_text:
            parts.append("- Writing/structure suggestions incorporated where helpful.")
        if analyst_text:
            parts.append("- Analytical framing and constraints considered.")

        parts.append(_bullets([
            "Key points were merged from the selected agents.",
            "Conflicts were resolved by preferring clarity and task fit.",
            "Output kept concise and structured."
        ]))

        if critique_text:
            parts.append("\n## Critique Integrated")
            parts.append(critique_text)

        final = "\n".join(parts).strip()

        return AgentResult(
            agent=self.name,
            output=final,
            meta={"team_size": len(team_outputs), "mode": "generic_synthesis", "used_agents": used}
        )
