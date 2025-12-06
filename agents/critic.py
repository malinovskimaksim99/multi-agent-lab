from typing import Dict, Any, List

class Critic:
    def review(self, draft: str) -> str:
        notes = self._notes(draft)
        return "\n".join(f"- {n}" for n in notes)

    def review_structured(self, draft: str) -> Dict[str, Any]:
        notes = self._notes(draft)
        tags: List[str] = []
        text = draft.lower()

        if "structured" not in text:
            tags.append("structure")
        if len(draft) < 200:
            tags.append("too_short")

        return {"notes": notes, "tags": tags}

    def _notes(self, draft: str):
        notes = []
        if "structured" not in draft.lower():
            notes.append("Add clearer structure (headings/bullets).")
        if len(draft) < 200:
            notes.append("Too short; add a bit more concrete steps.")
        if not notes:
            notes.append("Looks ok. Minor polish only.")
        return notes
