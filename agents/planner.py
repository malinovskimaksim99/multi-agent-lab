from typing import List, Optional
from .base import BaseAgent, AgentResult, Context, Memory
from .registry import register_agent


class Planner:
    def plan(self, task: str) -> List[str]:
        return [
            "Understand the task and expected output format",
            "Extract key points and constraints",
            "Draft an answer",
            "Run self-critique",
            "Revise and finalize",
        ]


@register_agent
class PlannerAgent(BaseAgent):
    name = "planner"
    description = "Creates a short step-by-step plan for solving the task."

    def __init__(self):
        self._planner = Planner()

    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        return 0.6

    def run(self, task: str, memory: Memory, context: Optional[Context] = None) -> AgentResult:
        plan = self._planner.plan(task)
        return AgentResult(agent=self.name, output=plan, meta={})
