from typing import Dict, Any, List, Optional, Tuple

# Ensure agents are registered
from . import planner as _planner  # noqa: F401
from . import analyst as _analyst  # noqa: F401
from . import critic as _critic    # noqa: F401
from . import writer as _writer    # noqa: F401
from . import synthesizer as _synth  # noqa: F401
from . import explainer as _explainer  # noqa: F401
from . import coder as _coder      # noqa: F401  # новий імпорт, щоб CoderAgent точно реєструвався

from .registry import create_agent, list_agents
from .base import Context, Memory
from .router import rank_agents


TEAM_PROFILES: Dict[str, List[str]] = {
    # документація / README — спочатку спеціалізований docs-агент, потім writer і analyst
    "docs": ["docs", "writer", "analyst"],
    # пояснення / огляди
    "explain": ["explainer", "analyst"],
    # планування / roadmap
    "planning": ["analyst", "explainer"],
    # код / помилки
    "code": ["coder", "analyst"],
}

# Task types where we definitely want automatic Critic passes.
# For other task types we can optionally skip Critic to save tokens.
CRITIC_TASK_TYPES = {
    "plan",
    "explain",
    "docs",
    "code",
    "db_analysis",
    "meta",
}

# Try to reuse task_type inference from router if available,
# otherwise fall back to a simple default.
try:
    from . import router as _router  # type: ignore
    infer_task_type_from_router = getattr(_router, "infer_task_type", lambda task: "other")
except Exception:
    def infer_task_type_from_router(task: str) -> str:
        return "other"


def infer_team_profile(task: str) -> str:
    t = task.lower()

    docs_markers = [
        "readme", "documentation", "docs", "guide", "installation",
        "інсталяц", "встанов", "документац", "гайд",
    ]
    if any(m in t for m in docs_markers):
        return "docs"

    # Код / помилки / traceback
    code_markers = [
        "traceback",
        "exception",
        "error",
        "bug",
        "fix",
        "refactor",
        "python",
        "javascript",
        "typescript",
        "sql",
        "api",
        "код",
        "скрипт",
        "помилка",
        "виправ",
    ]
    if any(m in t for m in code_markers):
        return "code"

    explain_markers = [
        "explain", "difference", "compare", "why", "how", "vs", "versus",
        "summary", "summarize", "overview", "roles of",
        "поясни", "різниц", "порівняй", "чому", "як працює", "підсумуй", "огляд",
    ]
    if any(m in t for m in explain_markers):
        return "explain"

    planning_markers = [
        "plan", "planning", "strategy", "roadmap", "outline",
        "план", "сплануй", "стратег", "роадмап", "дорожня карта",
    ]
    if any(m in t for m in planning_markers):
        return "planning"

    return "general"


class Supervisor:
    """
    Orchestrator.

    Modes:
      - single-solver:
          planner -> solver(auto or fixed) -> critic -> solver(revise)
      - team-solver:
          planner -> team solvers -> synthesizer(prelim) -> critic -> synthesizer(final)

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
        # не включаємо сюди coder — він має бути доступний як solver
        self.solver_exclude = set(solver_exclude or ["planner", "critic", "synthesizer"])

    def _pick_solver(self, task: str, memory: Memory, context: Context) -> str:
        """
        Вибір одного solver-агента для single-mode.

        Логіка:
          1) Якщо auto_solver=False -> повертаємо фіксований solver_name (звичайно 'analyst').
          2) Якщо задача схожа на кодову (traceback/error/python/код/помилка) і є coder ->
             повертаємо 'coder'.
          3) Інакше використовуємо загальний rank_agents.
        """
        if not self.auto_solver:
            return self.solver_name

        t = task.lower()
        available = set(list_agents())

        # Fast-path для кодових задач
        code_markers = [
            "traceback",
            "exception",
            "error",
            "bug",
            "fix",
            "refactor",
            "python",
            "javascript",
            "typescript",
            "sql",
            "api",
            "код",
            "скрипт",
            "помилка",
            "виправ",
        ]
        if any(m in t for m in code_markers):
            if "coder" in available and "coder" not in self.solver_exclude:
                return "coder"

        # Загальний ранжувальник
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
        # Визначаємо тип задачі один раз для всього пайплайну
        try:
            task_type = infer_task_type_from_router(task)
        except Exception:
            task_type = "other"

        context: Context = {}
        context["task_type"] = task_type

        # чи потрібно запускати Critic для цієї задачі
        need_critic = task_type in CRITIC_TASK_TYPES

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

            # Raw team draft (для налагодження)
            team_draft = "\n\n".join(
                [f"## {name} draft\n{out}" for name, out in team_outputs.items()]
            ).strip()

            # 3) PRELIM SYNTHESIS (перед критикою)
            synthesizer = create_agent(self.synthesizer_name)
            prelim_res = synthesizer.run(task, memory, context)
            prelim = prelim_res.output if isinstance(prelim_res.output, str) else team_draft

            # Critic має переглядати саме prelim
            context["draft"] = prelim

            # 4) Critique (авто-запуск для важливих task_type)
            tags: List[str] = []
            if need_critic:
                crit_res = critic.run(task, memory, context)
                crit_data = crit_res.output or {}
                notes = crit_data.get("notes", [])
                tags = crit_data.get("tags", [])

                if isinstance(notes, list):
                    critique_text = "\n".join(f"- {n}" for n in notes)
                else:
                    critique_text = str(notes) if notes else "- Looks ok."
            else:
                critique_text = f"- Auto-skip Critic for non-critical task_type='{task_type}'."

            context["critique_text"] = critique_text

            # 5) FINAL SYNTHESIS (з урахуванням critique_text)
            final_res = synthesizer.run(task, memory, context)
            final = final_res.output if isinstance(final_res.output, str) else prelim

            return {
                "task": task,
                "task_type": task_type,
                "plan": plan,
                "team_agents": team_names,
                "team_profile": profile,
                "solver_agent": self.synthesizer_name,
                "team_draft": team_draft,
                "draft": prelim,
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

        # 4) Critique (авто-запуск для важливих task_type)
        tags: List[str] = []
        if need_critic:
            crit_res = critic.run(task, memory, context)
            crit_data = crit_res.output or {}
            notes = crit_data.get("notes", [])
            tags = crit_data.get("tags", [])

            if isinstance(notes, list):
                critique_text = "\n".join(f"- {n}" for n in notes)
            else:
                critique_text = str(notes) if notes else "- Looks ok."
        else:
            critique_text = f"- Auto-skip Critic for non-critical task_type='{task_type}'."

        # 5) Revise if supported
        revise_fn = getattr(solver, "revise", None)
        final = revise_fn(draft, critique_text) if callable(revise_fn) else (draft + "\n\n" + critique_text)

        return {
            "task": task,
            "task_type": task_type,
            "plan": plan,
            "solver_agent": solver_name,
            "draft": draft,
            "critique": critique_text,
            "critique_tags": tags,
            "final": final,
            "available_agents": list_agents(),
        }