"""Depth-first solve over goal nodes."""

from typing import Any, Dict, List, Optional

import config
from logic.logic import AtomicFormula
from prolog.knowledge_base import KnowledgeBase, SoftKnowledgeBase
from solve.goalnode import GoalNode
from solve.search_guidance_policy import SearchGuidancePolicy, TrivialSearchGuidancePolicy


class DFSSolver:
    """Search for a proof using depth-first search with iterative deepening."""

    def __init__(
        self,
        kb: KnowledgeBase | SoftKnowledgeBase,
        max_depth: Optional[int] = None,
        search_guidance_policy: Optional[SearchGuidancePolicy] = None,
    ):
        self.kb = kb if isinstance(kb, SoftKnowledgeBase) else SoftKnowledgeBase(kb)
        self.max_depth = max_depth or config.DEFAULT_MAX_DEPTH
        self.search_guidance_policy = search_guidance_policy or TrivialSearchGuidancePolicy()

    def _filter_out_visited_nodes(self, nodes: List[GoalNode], visited: set) -> List[GoalNode]:
        unvisited = []
        for node in nodes:
            signature = node.signature()
            if signature in visited:
                continue
            visited.add(signature)
            unvisited.append(node)
        return unvisited

    def _dfs(self, node: GoalNode, goal,
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

        unvisited_successors = self._filter_out_visited_nodes(successors, visited)

        print(f"node {node} has unvisited successors: {unvisited_successors}")

        ordered_successors = self.search_guidance_policy.order_goals(
            goal,
            self.kb,
            node,
            min_confidence,
            unvisited_successors,
        )

        print(f"sorted {ordered_successors}")

        for successor in ordered_successors:
            proof = self._dfs(successor,goal, depth_limit=depth_limit,min_confidence=min_confidence,visited=visited,)
            if proof is not None:
                return proof

        self.kb, new_soft_facts, extended = self.search_guidance_policy.extend_on_backtrack(
            goal,
            node,
            min_confidence,
            self.kb,
        )
        print(f"new soft facts {new_soft_facts}")

        if not extended:
            return None

        successors = []
        for soft_fact in new_soft_facts:
            successors.extend(node.unify_soft_fact(soft_fact, min_confidence=min_confidence))
        unvisited_successors = self._filter_out_visited_nodes(successors, visited)

        ordered_successors = self.search_guidance_policy.order_goals(
            goal,
            self.kb,
            node,
            min_confidence,
            unvisited_successors,
        )

        for successor in ordered_successors:
            proof = self._dfs(successor,goal,depth_limit=depth_limit,min_confidence=min_confidence,visited=visited,)
            if proof is not None:
                return proof

        return None

    def solve(self, goal: AtomicFormula, min_confidence: float = 0.0) -> Dict[str, Any]:
        print(f"Solving with setting min_confidence={min_confidence}, depth_limit={self.max_depth}")
        """Run iterative-deepening DFS and return the first proven node found."""
        root = GoalNode(formulas=[goal], depth=0, confidence=1.0)

        visited = {root.signature()}
        proof = self._dfs(root, goal, depth_limit=self.max_depth + 1, min_confidence=min_confidence, visited=visited)
        if proof is not None:
            return {
                "success": True,
                "proof": proof,
                "confidence": proof.is_proven(),
            }

        return {
            "success": False,
            "proof": None,
            "confidence": 0.0,
        }
