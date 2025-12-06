from typing import List, Dict, Any

class Analyst:
    def draft(self, task: str, plan: List[str], memory: Dict[str, Any]) -> str:
        flags = memory.get("flags", {}) or {}
        force_structure = bool(flags.get("force_structure"))

        base_points = [
            "Identify the core request and constraints.",
            "Provide a concise answer with actionable points.",
            "Include a quick self-check for gaps."
        ]

        if force_structure:
            body = (
                "## Task\n"
                f"{task}\n\n"
                "## Plan\n" +
                "\n".join(f"{i+1}. {p}" for i, p in enumerate(plan)) +
                "\n\n## Answer\n" +
                "\n".join(f"- {p}" for p in base_points)
            )
        else:
            body = (
                f"Task: {task}\n"
                "Plan: " + ", ".join(plan) + "\n"
                "Answer: " + " ".join(base_points)
            )

        return body

    def revise(self, draft: str, critique: str) -> str:
        # Simple revision pass
        return draft + "\n\n## Revisions\n" + critique
