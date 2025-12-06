from typing import Dict, Any, List

class Critic:
    def review(self, draft: str) -> str:
        notes = self._notes(draft)
        return "\n".join(f"- {n}" for n in notes)

    def review_structured(self, draft: str) -> Dict[str, Any]:
        notes = self._notes(draft)
        tags: List[str] = []

        if not self._has_structure(draft):
            tags.append("structure")

        if len(draft.strip()) < 200:
            tags.append("too_short")

        return {"notes": notes, "tags": tags}

    def _has_structure(self, text: str) -> bool:
        t = text
        return (
            "##" in t or
            "\n-" in t or
            "\n1." in t or
            "\n2." in t
        )

    def _notes(self, draft: str):
        notes = []
        if not self._has_structure(draft):
            notes.append("Add clearer structure (headings/bullets).")
        if len(draft.strip()) < 200:
            notes.append("Too short; add a bit more concrete steps.")
        if not notes:
            notes.append("Looks ok. Minor polish only.")
        return notes
