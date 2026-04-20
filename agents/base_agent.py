"""Base Agent Class for Multi-Agent Music System."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path


class BaseAgent(ABC):
    """Abstract base class for all music generation agents."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary function."""
        pass

    def get_cost_estimate(self, input_data: Dict[str, Any]) -> float:
        """Estimate cost of executing this agent."""
        return 0.0

    def log(self, message: str):
        """Log agent activity."""
        print(f"[{self.name}] {message}")