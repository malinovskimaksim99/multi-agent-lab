from typing import Dict, Any, List, Optional

# Ensure agents are registered
from . import planner as _planner  # noqa: F401
from . import analyst as _analyst  # noqa: F401
from . import critic as _critic    # noqa: F401
from . import writer as _writer    # noqa: F401
from . import synthesizer as _synth  # noqa: F401

from .registry import create_agent, list_agents
from .base import Context, Memory
from .router import rank_agents


class Supervisor:
    """
    Orchestrator.

    Modes:
      - single-solver:
          planner -> solver(auto or fixed) -> critic -> solver(revise)
      - team-solver:
          planner -> top-K solvers -> critic -> synthesizer

    Team mode is designed to let specialized agents collaborate
    (e.g., writer + analyst) while keeping planner/critic roles stable.
    """

    def __init__(
        self,
        planner_name: str = "planner",
        critic_name: str = "critic",
        solver_name: str = "analyst",
        synthesizer_name: str = "synthesizer",
        auto_solver: bool = False,
        auto_team: bool = False,
        team_size: int = 2,
        solver_exclude: Optional[List[str]] = None,
    ):
        self.planner_name = planner_name
        self.critic_name = critic_name
        self.solver_name = solver_name
        self.synthesizer_name = synthesizer_name
        self.auto_solver = auto_solver
        self.auto_team = auto_team
        self.team_size = max(1, int(team_size))
        self.solver_exclude = set(solver_exclude or ["planner", "critic", "synthesizer"])

    def _pick_solver(self, task: str, memory: Memory, context: Context) -> str:
        if not self.auto_solver:
            return self.solver_name

        ranked = rank_agents(task, memory, context)
        for name, score in ranked:
            if name in self.solver_exclude:
                continue
            if score > 0:
                return name

        return self.solver_name

    def _pick_team(self, task: str, memory: Memory, context: Context) -> List[str]:
        ranked = rank_agents(task, memory, context)

        team: List[str] = []
        for name, score in ranked:
            if name in self.solver_exclude:
                continue
            if score <= 0:
                continue
            team.append(name)
            if len(team) >= self.team_size:
                break

        if not team:
            team = [self.solver_name]

        return team

    def run(self, task: str, memory: Memory) -> Dict[str, Any]:
        context: Context = {}

        planner = create_agent(self.planner_name)
        critic = create_agent(self.critic_name)

        # 1) Plan
        plan_res = planner.run(task, memory, context)
        plan = plan_res.output if isinstance(plan_res.output, list) else []
        context["plan"] = plan

        # 2) TEAM MODE
        if self.auto_team:
            team_names = self._pick_team(task, memory, context)
            team_outputs: Dict[str, str] = {}

            for name in team_names:
                agent = create_agent(name)
                res = agent.run(task, memory, context)
                out = res.output if isinstance(res.output, str) else str(res.output)
                team_outputs[name] = out

            context["team_outputs"] = team_outputs

            # Provide a combined draft for the critic
            team_draft = "\n\n".join(
                [f"## {name} draft\n{out}" for name, out in team_outputs.items()]
            ).strip()
            context["draft"] = team_draft

            # 3) Critique
            crit_res = critic.run(task, memory, context)
            crit_data = crit_res.output or {}
            notes = crit_data.get("notes", [])
            tags = crit_data.get("tags", [])

            if isinstance(notes, list):
                critique_text = "\n".join(f"- {n}" for n in notes)
            else:
                critique_text = str(notes) if notes else "- Looks ok."

            context["critique_text"] = critique_text

            # 4) Synthesize
            synthesizer = create_agent(self.synthesizer_name)
            syn_res = synthesizer.run(task, memory, context)
            final = syn_res.output if isinstance(syn_res.output, str) else team_draft

            return {
                "task": task,
                "plan": plan,
                "team_agents": team_names,
                "solver_agent": self.synthesizer_name,
                "draft": team_draft,
                "critique": critique_text,
                "critique_tags": tags,
                "final": final,
                "available_agents": list_agents(),
            }

        # 2) SINGLE SOLVER MODE
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

        # 5) Revise if supported
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
