from typing import List

class Planner:
    def plan(self, task: str) -> List[str]:
        return [
            "Understand the task and expected output format",
            "Extract key points and constraints",
            "Draft an answer",
            "Run self-critique",
            "Revise and finalize",
        ]
