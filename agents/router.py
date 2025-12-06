from typing import List, Optional, Tuple
from .registry import list_agents, create_agent
from .base import Context, Memory


def rank_agents(
    task: str,
    memory: Memory,
    context: Optional[Context] = None
) -> List[Tuple[str, float]]:
    scores = []
    for name in list_agents():
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
    context: Optional[Context] = None
) -> List[str]:
    ranked = rank_agents(task, memory, context)
    return [name for name, score in ranked[:k] if score > 0]
