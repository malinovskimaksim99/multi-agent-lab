from typing import Dict, Any, List, Optional, Tuple

# Ensure agents are registered
from . import planner as _planner  # noqa: F401
from . import analyst as _analyst  # noqa: F401
from . import critic as _critic    # noqa: F401
from . import writer as _writer    # noqa: F401
from . import synthesizer as _synth  # noqa: F401
from . import explainer as _explainer  # noqa: F401

from .registry import create_agent, list_agents
from .base import Context, Memory
from .router import rank_agents


TEAM_PROFILES: Dict[str, List[str]] = {
    "docs": ["writer", "analyst"],
    "explain": ["explainer", "analyst"],
    "planning": ["analyst", "explainer"],
    # placeholder until we add a dedicated coder agent
    "code": ["analyst"],
}


def infer_team_profile(task: str) -> str:
    t = task.lower()

    # docs
    docs_markers = [
        "readme", "documentation", "docs", "guide", "installation",
        "інсталяц", "встанов", "документац", "гайд"
    ]
    if any(m in t for m in docs_markers):
        return "docs"

    # explain
    explain_markers = [
        "explain", "difference", "compare", "why", "how", "vs", "versus",
        "поясни", "різниц", "порівняй", "чому", "як працює", "що таке"
    ]
    if any(m in t for m in explain_markers):
        return "explain"

    # planning
    planning_markers = [
        "plan", "planning", "strategy", "roadmap", "outline",
        "план", "сплануй", "стратег", "роадмап", "дорожня карта"
    ]
    if any(m in t for m in planning_markers):
        return "planning"

    # code
    code_markers = [
        "code", "bug", "error", "fix", "refactor",
        "python", "javascript", "typescript", "sql", "api",
        "код", "скрипт", "помилка", "виправ"
    ]
    if any(m in t for m in code_markers):
        return "code"

    return "general"


class Supervisor:
    """
    Orchestrator.

    Modes:
      - single-solver:
          planner -> solver(auto or fixed) -> critic -> solver(revise)
      - team-solver:
          planner -> team solvers -> critic -> synthesizer

    Team profiles:
      When enabled, we seed the team with profile-preferred agents
      before filling remaining slots using router ranking.
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
        use_team_profiles: bool = True,
        solver_exclude: Optional[List[str]] = None,
    ):
        self.planner_name = planner_name
        self.critic_name = critic_name
        self.solver_name = solver_name
        self.synthesizer_name = synthesizer_name
        self.auto_solver = auto_solver
        self.auto_team = auto_team
        self.team_size = max(1, int(team_size))
        self.use_team_profiles = use_team_profiles
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

    def _seed_team_by_profile(self, profile: str) -> List[str]:
        preferred = TEAM_PROFILES.get(profile, [])
        available = set(list_agents())
        team: List[str] = []
        for name in preferred:
            if name in self.solver_exclude:
                continue
            if name not in available:
                continue
            if name not in team:
                team.append(name)
        return team

    def _pick_team(self, task: str, memory: Memory, context: Context) -> Tuple[List[str], str]:
        profile = infer_team_profile(task) if self.use_team_profiles else "general"

        team: List[str] = []
        if profile != "general":
            team = self._seed_team_by_profile(profile)

        if len(team) < self.team_size:
            ranked = rank_agents(task, memory, context)
            for name, score in ranked:
                if name in self.solver_exclude or name in team:
                    continue
                if score <= 0:
                    continue
                team.append(name)
                if len(team) >= self.team_size:
                    break

        if not team:
            team = [self.solver_name]

        return team[: self.team_size], profile

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
            team_names, profile = self._pick_team(task, memory, context)
            team_outputs: Dict[str, str] = {}

            for name in team_names:
                agent = create_agent(name)
                res = agent.run(task, memory, context)
                out = res.output if isinstance(res.output, str) else str(res.output)
                team_outputs[name] = out

            context["team_outputs"] = team_outputs

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
                "team_profile": profile,
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
