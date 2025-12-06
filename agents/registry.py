from typing import Dict, Type, List
from .base import BaseAgent

_REGISTRY: Dict[str, Type[BaseAgent]] = {}


def register_agent(cls: Type[BaseAgent]) -> Type[BaseAgent]:
    name = getattr(cls, "name", cls.__name__).lower()
    if not name:
        raise ValueError("Agent must have a non-empty name")
    _REGISTRY[name] = cls
    return cls


def list_agents() -> List[str]:
    return sorted(_REGISTRY.keys())


def get_agent_class(name: str) -> Type[BaseAgent]:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Agent '{name}' not found. Available: {list_agents()}")
    return _REGISTRY[key]


def create_agent(name: str, **kwargs) -> BaseAgent:
    cls = get_agent_class(name)
    return cls(**kwargs)
