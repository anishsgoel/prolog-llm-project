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
        self.failed_atoms: List[AtomicFormula] = []

    def _make_root_goal(self, goal: AtomicFormula) -> GoalNode:
        return GoalNode(formulas=[goal], depth=0, confidence=1.0)

    def _push_node(self, queue: list, tie: count, node: GoalNode) -> None:
        proof_confidence = node.is_proven()
        priority_confidence = proof_confidence if proof_confidence > 0.0 else node.confidence
        heapq.heappush(queue, (-priority_confidence, node.depth, next(tie), node))

    def _node_signature(self, node: GoalNode) -> tuple:
        return node.signature()

    def _record_failed_atoms(self, atoms: List[AtomicFormula]) -> None:
        print(f"    failed atoms : {atoms}", file=open("unifications.txt", "a"))
        self.failed_atoms.extend(atoms)

    def solve(self, goal: AtomicFormula, min_confidence: float = 0.0) -> Dict[str, Any]:
        """Run the solve and return the first proven node found."""
        root = self._make_root_goal(goal)
        self.failed_atoms = []

        queue: List[tuple] = []
        tie = count()
        self._push_node(queue, tie, root)

        visited = set()
        visited.add(self._node_signature(root))

        while queue:
            _, _, _, node = heapq.heappop(queue)

            if node.depth > self.max_depth:
                continue

            node.mark_proved_facts(self.kb, min_confidence=min_confidence)
            proof_confidence = node.is_proven()
            if proof_confidence > 0.0:
                return {
                    "success": True,
                    "proof": node,
                    "confidence": proof_confidence,
                    "failed_atoms": self.failed_atoms,
                }

            if node.depth >= self.max_depth:
                continue

            successors = []
            successors.extend(node.unify_soft_kb(self.kb, min_confidence=min_confidence))
            successors.extend(node.unify_soft_rules(self.kb, min_confidence=min_confidence))

            print("***********", file=open("unifications.txt", "a"))
            print(f"node: {node}", file=open("unifications.txt", "a"))

            has_successor = False
            for successor in successors:
                signature = self._node_signature(successor)
                if signature not in visited:
                    print(f"    successor: f{successor}", file=open("unifications.txt", "a"))
                    self._push_node(queue, tie, successor)
                    visited.add(signature)
                    has_successor = True

            if not has_successor:
                self._record_failed_atoms(node.unresolved_formulas())

        return {
            "success": False,
            "proof": None,
            "confidence": 0.0,
            "failed_atoms": self.failed_atoms,
        }
