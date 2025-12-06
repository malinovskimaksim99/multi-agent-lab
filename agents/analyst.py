from typing import List, Dict, Any

class Analyst:
    def draft(self, task: str, plan: List[str], memory: Dict[str, Any]) -> str:
        rules = memory.get("rules", [])
        rules_text = "\n".join(f"- {r}" for r in rules) if rules else "- (no stored rules yet)"
        return (
            f"Task:\n{task}\n\n"
            f"Plan:\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(plan)) + "\n\n"
            f"Applied memory rules:\n{rules_text}\n\n"
            "Draft answer:\n"
            "Here is a concise, structured response based on the plan."
        )

    def revise(self, draft: str, critique: str) -> str:
        return draft + "\n\nRevisions applied based on critique:\n" + critique
