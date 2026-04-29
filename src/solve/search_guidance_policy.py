"""Interfaces for guiding DFS proof search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from prolog.knowledge_base import SoftKnowledgeBase
from prolog.prolog_command import SoftFact
from solve.goalnode import GoalNode
from typing import Tuple


class SearchGuidancePolicy(ABC):
    """Controls goal ordering and KB extension during DFS backtracking."""

    @abstractmethod
    def order_goals(
        self,
        soft_kb: SoftKnowledgeBase,
        current_goal_node: GoalNode,
        goal_nodes: List[GoalNode],
    ) -> List[GoalNode]:
        """Return ``goal_nodes`` reordered for the next DFS step."""

    @abstractmethod
    def extend_on_backtrack(
        self,
        goal_node: GoalNode,
        soft_kb: SoftKnowledgeBase,
    ) -> Tuple[SoftKnowledgeBase, list[SoftFact], bool]:
        """Return an extended KB for backtracking."""


class TrivialSearchGuidancePolicy(SearchGuidancePolicy):
    """Default policy that preserves the given order and never extends the KB."""

    def order_goals(
        self,
        soft_kb: SoftKnowledgeBase,
        current_goal_node: GoalNode,
        goal_nodes: List[GoalNode],
    ) -> List[GoalNode]:
        return goal_nodes

    def extend_on_backtrack(
        self,
        goal_node: GoalNode,
        soft_kb: SoftKnowledgeBase,
    ) -> Tuple[SoftKnowledgeBase, list[SoftFact], bool]:
        return soft_kb, [], False
