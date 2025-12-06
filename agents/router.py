from typing import List, Optional, Tuple, Set
from .registry import list_agents, create_agent
from .base import Context, Memory

DEFAULT_EXCLUDE: Set[str] = {"planner", "critic"}


def rank_agents(
    task: str,
    memory: Memory,
    context: Optional[Context] = None,
    exclude: Optional[Set[str]] = None
) -> List[Tuple[str, float]]:
    ex = set(DEFAULT_EXCLUDE)
    if exclude:
        ex |= set(exclude)

    scores: List[Tuple[str, float]] = []
    for name in list_agents():
        if name in ex:
            continue

        agent = create_agent(name)
        try:
            score = float(agent.can_handle(task, context))
        except Exception:
            score = 0.0

        scores.append((name, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def pick_top(
    task: str,
    memory: Memory,
    k: int = 2,
    context: Optional[Context] = None,
    exclude: Optional[Set[str]] = None
) -> List[str]:
    ranked = rank_agents(task, memory, context=context, exclude=exclude)
    return [name for name, score in ranked[:k] if score > 0]
