"""Priority-based solve over goal nodes."""

import heapq
from itertools import count
from typing import Any, Dict, List, Optional

import config
from logic.logic import AtomicFormula
from prolog.knowledge_base import KnowledgeBase, SoftKnowledgeBase
from solve.goalnode import GoalNode


class Solver:
    """Search for a proof using a confidence-prioritized queue of goal nodes."""

    def __init__(self, kb: KnowledgeBase | SoftKnowledgeBase, max_depth: Optional[int] = None):
        self.kb = kb if isinstance(kb, SoftKnowledgeBase) else SoftKnowledgeBase(kb)
        self.max_depth = max_depth or config.DEFAULT_MAX_DEPTH
        self.failed_goals: List[GoalNode] = []

    def _make_root_goal(self, goal: AtomicFormula) -> GoalNode:
        return GoalNode(formulas=[goal], depth=0, confidence=1.0)

    def _push_node(self, queue: list, tie: count, node: GoalNode) -> None:
        proof_confidence = node.is_proven()
        priority_confidence = proof_confidence if proof_confidence > 0.0 else node.confidence
        heapq.heappush(queue, (-priority_confidence, node.depth, next(tie), node))

    def _node_signature(self, node: GoalNode) -> tuple:
        return node.signature()

    def solve(self, goal: AtomicFormula, min_confidence: float = 0.0) -> Dict[str, Any]:
        """Run the solve and return the first proven node found."""
        root = self._make_root_goal(goal)
        self.failed_goals = []

        queue: List[tuple] = []
        tie = count()
        self._push_node(queue, tie, root)

        visited = set()

        while queue:
            _, _, _, node = heapq.heappop(queue)

            if node.depth > self.max_depth:
                self.failed_goals.append(node)
                continue

            signature = self._node_signature(node)
            if signature in visited:
                continue
            visited.add(signature)

            node.mark_proved_facts(self.kb, min_confidence=min_confidence)
            proof_confidence = node.is_proven()
            if proof_confidence > 0.0:
                return {
                    "success": True,
                    "proof": node,
                    "confidence": proof_confidence,
                    "failed_goals": self.failed_goals,
                }

            successors = []
            successors.extend(node.unify_soft_kb(self.kb, min_confidence=min_confidence))
            successors.extend(node.unify_soft_rules(self.kb, min_confidence=min_confidence))

            if not successors:
                self.failed_goals.append(node)
                continue

            for successor in successors:
                if successor.depth <= self.max_depth:
                    self._push_node(queue, tie, successor)
                else:
                    self.failed_goals.append(successor)

        return {
            "success": False,
            "proof": None,
            "confidence": 0.0,
            "failed_goals": self.failed_goals,
        }
