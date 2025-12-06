from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

Context = Dict[str, Any]
Memory = Dict[str, Any]


@dataclass
class AgentResult:
    agent: str
    output: Any
    meta: Dict[str, Any]


class BaseAgent(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def can_handle(self, task: str, context: Optional[Context] = None) -> float:
        """Return 0.0..1.0 how suitable this agent is for the task."""
        ...

    @abstractmethod
    def run(
        self,
        task: str,
        memory: Memory,
        context: Optional[Context] = None
    ) -> AgentResult:
        ...
