"""Shared abstractions for LLM-based DFS search guidance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

from solve.goalnode import GoalNode


@dataclass(frozen=True)
class LLMSearchGuidancePromptContext:
    """Context exposed to LLM search-guidance prompt builders."""

    goal: str
    current_goal_lines: List[str]
    hard_fact_lines: List[str]
    hard_rule_lines: List[str]
    soft_fact_lines: List[str]
    soft_rule_lines: List[str]
    predicate_comments: Dict[str, str]
    min_confidence: float


class PromptBuilder(ABC):
    """Abstract prompt provider for LLM-driven DFS guidance."""

    @abstractmethod
    def order_schema(self) -> str:
        """Return the repair schema for goal ordering."""

    @abstractmethod
    def backtrack_schema(self) -> str:
        """Return the repair schema for backtracking extension."""

    @abstractmethod
    def order_prompt(
        self,
        context: LLMSearchGuidancePromptContext,
        goal_nodes: List[GoalNode],
    ) -> str:
        """Build the prompt used to order DFS successor nodes."""

    @abstractmethod
    def backtrack_prompt(
        self,
        context: LLMSearchGuidancePromptContext,
    ) -> str:
        """Build the prompt used to extend the soft KB on backtracking."""
