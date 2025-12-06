from typing import Dict, Any, List, Optional

# важливо: імпорти нижче гарантують реєстрацію агентів
from . import planner as _planner  # noqa: F401
from . import analyst as _analyst  # noqa: F401
from . import critic as _critic    # noqa: F401
from . import writer as _writer    # noqa: F401

from .registry import create_agent, list_agents
from .base import Context, Memory
from .router import rank_agents


class Supervisor:
    """
    Orchestrator.

    Default pipeline:
      planner -> solver(auto or fixed) -> critic -> solver(revise)

    - planner is always used
    - critic is always used
    - solver can be:
        * fixed (analyst by default)
        * auto-selected from registry using router scores
    """

    def __init__(
        self,
        planner_name: str = "planner",
        critic_name: str = "critic",
        solver_name: str = "analyst",
        auto_solver: bool = False,
        solver_exclude: Optional[List[str]] = None,
    ):
        self.planner_name = planner_name
        self.critic_name = critic_name
        self.solver_name = solver_name
        self.auto_solver = auto_solver
        self.solver_exclude = set(solver_exclude or ["planner", "critic"])

    def _pick_solver(self, task: str, memory: Memory, context: Context) -> str:
        if not self.auto_solver:
            return self.solver_name

        ranked = rank_agents(task, memory, context)

        # filter out planner/critic and any excluded agents
        for name, score in ranked:
            if name in self.solver_exclude:
                continue
            if score > 0:
                return name

        # fallback
        return self.solver_name

    def run(self, task: str, memory: Memory) -> Dict[str, Any]:
        context: Context = {}

        planner = create_agent(self.planner_name)
        critic = create_agent(self.critic_name)

        # 1) Plan
        plan_res = planner.run(task, memory, context)
        plan = plan_res.output if isinstance(plan_res.output, list) else []
        context["plan"] = plan

        # 2) Pick solver (auto or fixed)
        solver_name = self._pick_solver(task, memory, context)
        solver = create_agent(solver_name)

        # 3) Draft
        draft_res = solver.run(task, memory, context)
        draft = draft_res.output if isinstance(draft_res.output, str) else str(draft_res.output)
        context["draft"] = draft

        # 4) Critique
        crit_res = critic.run(task, memory, context)
        crit_data = crit_res.output or {}
        notes = crit_data.get("notes", [])
        tags = crit_data.get("tags", [])

        if isinstance(notes, list):
            critique_text = "\n".join(f"- {n}" for n in notes)
        else:
            critique_text = str(notes) if notes else "- Looks ok."

        # 5) Revise if solver supports it
        revise_fn = getattr(solver, "revise", None)
        final = revise_fn(draft, critique_text) if callable(revise_fn) else (draft + "\n\n" + critique_text)

        return {
            "task": task,
            "plan": plan,
            "solver_agent": solver_name,
            "draft": draft,
            "critique": critique_text,
            "critique_tags": tags,
            "final": final,
            "available_agents": list_agents(),
        }
