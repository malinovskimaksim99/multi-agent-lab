from typing import Dict, Any
from .planner import Planner
from .analyst import Analyst
from .critic import Critic

class Supervisor:
    def __init__(self):
        self.planner = Planner()
        self.analyst = Analyst()
        self.critic = Critic()

    def run(self, task: str, memory: Dict[str, Any]) -> Dict[str, Any]:
        plan = self.planner.plan(task)
        draft = self.analyst.draft(task, plan, memory)
        critique = self.critic.review(draft)
        final = self.analyst.revise(draft, critique)

        return {
            "task": task,
            "plan": plan,
            "draft": draft,
            "critique": critique,
            "final": final,
        }
