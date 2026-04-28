"""Depth-first solve over goal nodes."""

from typing import Any, Dict, List, Optional

import config
from logic.logic import AtomicFormula
from prolog.knowledge_base import KnowledgeBase, SoftKnowledgeBase
from solve.goalnode import GoalNode


class DFSSolver:
    """Search for a proof using depth-first search with iterative deepening."""

    def __init__(self, kb: KnowledgeBase | SoftKnowledgeBase, max_depth: Optional[int] = None):
        self.kb = kb if isinstance(kb, SoftKnowledgeBase) else SoftKnowledgeBase(kb)
        self.max_depth = max_depth or config.DEFAULT_MAX_DEPTH

    def _dfs(
        self,
        node: GoalNode,
        depth_limit: int,
        min_confidence: float,
        visited: set,
    ) -> GoalNode | None:
        if node.depth > depth_limit:
            return None

        node.mark_proved_facts(self.kb, min_confidence=min_confidence)
        if node.is_proven() > 0.0:
            return node

        if node.depth >= depth_limit:
            return None

        successors = []
        successors.extend(node.unify_soft_kb(self.kb, min_confidence=min_confidence))
        successors.extend(node.unify_soft_rules(self.kb, min_confidence=min_confidence))

        for successor in successors:
            signature = successor.signature()
            if signature in visited:
                continue

            has_unvisited_successor = True
            visited.add(signature)
            proof = self._dfs(successor, depth_limit=depth_limit, min_confidence=min_confidence, visited=visited)
            if proof is not None:
                return proof

        return None

    def solve(self, goal: AtomicFormula, min_confidence: float = 0.0) -> Dict[str, Any]:
        """Run iterative-deepening DFS and return the first proven node found."""
        root = GoalNode(formulas=[goal], depth=0, confidence=1.0)

        for depth_limit in range(root.depth, self.max_depth + 1):
            visited = {root.signature()}
            proof = self._dfs(root, depth_limit=depth_limit, min_confidence=min_confidence, visited=visited)
            if proof is not None:
                return {
                    "success": True,
                    "proof": proof,
                    "confidence": proof.is_proven(),
                    "failed_atoms": self.failed_atoms,
                }

        return {
            "success": False,
            "proof": None,
            "confidence": 0.0,
            "failed_atoms": self.failed_atoms,
        }
