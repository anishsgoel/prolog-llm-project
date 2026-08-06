"""Extension strategies for the meta-solver."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from logic.logic import AtomicFormula
from prolog.knowledge_base import SoftKnowledgeBase


class ExtensionStrategy(ABC):
    """Interface for extending the soft KB and search limits between solver runs."""

    @abstractmethod
    def extend(
        self,
        soft_kb: SoftKnowledgeBase,
        goal: AtomicFormula,
        failed_atoms: List[AtomicFormula],
        max_depth: int,
        min_confidence: float,
    ) -> Tuple[SoftKnowledgeBase, int, float]:
        """Return an updated soft KB, max depth, and confidence limit."""


class TrivialExtensionStrategy(ExtensionStrategy):
    """Minimal strategy that only increases the search depth by one."""

    def extend(
        self,
        soft_kb: SoftKnowledgeBase,
        goal: AtomicFormula,
        failed_atoms: List[AtomicFormula],
        max_depth: int,
        min_confidence: float,
    ) -> Tuple[SoftKnowledgeBase, int, float]:
        return soft_kb, max_depth + 1, min_confidence
