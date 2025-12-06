from typing import Dict, Any

# важливо: імпорти нижче гарантують реєстрацію агентів
from . import planner as _planner  # noqa: F401
from . import analyst as _analyst  # noqa: F401
from . import critic as _critic    # noqa: F401

from .registry import create_agent
from .base import Context, Memory


class Supervisor:
    """
    Orchestrator with a default pipeline:
    planner -> analyst -> critic -> analyst(revise)
    Uses the registry so you can swap/add agents later.
    """

    def __init__(
        self,
        planner_name: str = "planner",
        analyst_name: str = "analyst",
        critic_name: str = "critic",
    ):
        self.planner_name = planner_name
        self.analyst_name = analyst_name
        self.critic_name = critic_name

    def run(self, task: str, memory: Memory) -> Dict[str, Any]:
        context: Context = {}

        planner = create_agent(self.planner_name)
        analyst = create_agent(self.analyst_name)
        critic = create_agent(self.critic_name)

        plan_res = planner.run(task, memory, context)
        plan = plan_res.output
        context["plan"] = plan

        draft_res = analyst.run(task, memory, context)
        draft = draft_res.output
        context["draft"] = draft

        crit_res = critic.run(task, memory, context)
        crit_data = crit_res.output or {}
        notes = crit_data.get("notes", [])
        tags = crit_data.get("tags", [])

        critique_text = "\n".join(f"- {n}" for n in notes) if isinstance(notes, list) else str(notes)

        revise_fn = getattr(analyst, "revise", None)
        final = revise_fn(draft, critique_text) if callable(revise_fn) else (draft + "\n\n" + critique_text)

        return {
            "task": task,
            "plan": plan,
            "draft": draft,
            "critique": critique_text,
            "critique_tags": tags,
            "final": final,
        }
