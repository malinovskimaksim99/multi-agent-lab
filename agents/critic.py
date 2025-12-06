class Critic:
    def review(self, draft: str) -> str:
        notes = []
        if "structured" not in draft.lower():
            notes.append("Add clearer structure (headings/bullets).")
        if len(draft) < 200:
            notes.append("Too short; add a bit more concrete steps.")
        if not notes:
            notes.append("Looks ok. Minor polish only.")
        return "\n".join(f"- {n}" for n in notes)
