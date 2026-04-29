"""Meta-solver orchestration for the DFS solver."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import config
from logic.logic import AtomicFormula
from prolog.knowledge_base import KnowledgeBase, SoftKnowledgeBase
from solve.dfssolver import DFSSolver
from solve.search_guidance_policy import SearchGuidancePolicy


class DFSMetaSolver:
    """Repeatedly run the DFS solver while tuning depth and confidence threshold."""

    def __init__(
        self,
        kb: KnowledgeBase,
        search_guidance_policy: SearchGuidancePolicy,
        max_depth: Optional[int] = None,
        min_confidence: float = 1.0,
        max_depth_ceiling: Optional[int] = None,
        confidence_tolerance: float = 0.05,
        max_binary_search_steps: int = 8,
    ):
        self.hard_kb = kb
        self.soft_kb = SoftKnowledgeBase(kb)
        self.search_guidance_policy = search_guidance_policy
        self.max_depth = max_depth or config.DEFAULT_MAX_DEPTH
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self.max_depth_ceiling = max_depth_ceiling or (self.max_depth + config.DEFAULT_MAX_DEPTH)
        self.confidence_tolerance = max(confidence_tolerance, 1e-6)
        self.max_binary_search_steps = max(1, max_binary_search_steps)

    def _log_attempt(
        self,
        depth_limit: int,
        min_confidence: float,
        result: Dict[str, Any],
        old_soft_kb: SoftKnowledgeBase,
        new_soft_kb: SoftKnowledgeBase,
    ) -> None:
        if not config.VERBOSE:
            return

        print("[DFSMetaSolver] Attempt")
        print(
            "[DFSMetaSolver] Parameters: max_depth={}, min_confidence={:.3f}, success={}, confidence={:.3f}".format(
                depth_limit,
                min_confidence,
                result.get("success", False),
                result.get("confidence", 0.0),
            )
        )

        old_fact_nums = {fact.num for fact in old_soft_kb.facts}
        old_rule_nums = {rule.num for rule in old_soft_kb.rules}
        new_facts = [fact for fact in new_soft_kb.facts if fact.num not in old_fact_nums]
        new_rules = [rule for rule in new_soft_kb.rules if rule.num not in old_rule_nums]

        if new_facts:
            print("[DFSMetaSolver] New facts:")
            for fact in new_facts:
                print(f"  - {fact}")

        if new_rules:
            print("[DFSMetaSolver] New rules:")
            for rule in new_rules:
                print(f"  - {rule}")

        if not new_facts and not new_rules:
            print("[DFSMetaSolver] No new clauses added.")

    def _run_solver(
        self,
        goal: AtomicFormula,
        soft_kb: SoftKnowledgeBase,
        depth_limit: int,
        min_confidence: float,
    ) -> tuple[Dict[str, Any], SoftKnowledgeBase]:
        old_soft_kb = soft_kb.copy()
        solver = DFSSolver(
            soft_kb,
            max_depth=depth_limit,
            search_guidance_policy=self.search_guidance_policy,
        )
        result = solver.solve(goal, min_confidence=min_confidence)
        updated_soft_kb = solver.kb
        self._log_attempt(depth_limit, min_confidence, result, old_soft_kb, updated_soft_kb)
        return result, updated_soft_kb

    def _search_confidence(
        self,
        goal: AtomicFormula,
        soft_kb: SoftKnowledgeBase,
        depth_limit: int,
    ) -> tuple[Optional[Dict[str, Any]], SoftKnowledgeBase, float, List[Dict[str, Any]]]:
        low = 0.0
        high = 1.0
        probe = self.min_confidence
        tried = set()

        best_result: Optional[Dict[str, Any]] = None
        best_threshold = 0.0
        attempts: List[Dict[str, Any]] = []

        for _ in range(self.max_binary_search_steps):
            probe = max(low, min(high, probe))
            rounded_probe = round(probe, 6)
            if rounded_probe in tried:
                break
            tried.add(rounded_probe)

            result, soft_kb = self._run_solver(
                goal=goal,
                soft_kb=soft_kb,
                depth_limit=depth_limit,
                min_confidence=probe,
            )
            attempts.append(
                {
                    "max_depth": depth_limit,
                    "min_confidence": probe,
                    "success": result.get("success", False),
                    "confidence": result.get("confidence", 0.0),
                }
            )

            if result.get("success"):
                best_result = result
                best_threshold = probe
                low = probe
                if high - low <= self.confidence_tolerance:
                    break
                probe = (probe + high) / 2.0
            else:
                high = probe
                if high - low <= self.confidence_tolerance:
                    break
                probe = (low + probe) / 2.0

        if best_result is None and 0.0 not in tried:
            result, soft_kb = self._run_solver(
                goal=goal,
                soft_kb=soft_kb,
                depth_limit=depth_limit,
                min_confidence=0.0,
            )
            attempts.append(
                {
                    "max_depth": depth_limit,
                    "min_confidence": 0.0,
                    "success": result.get("success", False),
                    "confidence": result.get("confidence", 0.0),
                }
            )
            if result.get("success"):
                best_result = result
                best_threshold = 0.0

        return best_result, soft_kb, best_threshold, attempts

    def solve(self, goal: AtomicFormula) -> Dict[str, Any]:
        """Search over depth and confidence thresholds using DFSSolver."""
        soft_kb = self.soft_kb
        attempts: List[Dict[str, Any]] = []

        for depth_limit in range(self.max_depth, self.max_depth_ceiling + 1):
            best_result, soft_kb, best_threshold, depth_attempts = self._search_confidence(
                goal=goal,
                soft_kb=soft_kb,
                depth_limit=depth_limit,
            )
            attempts.extend(depth_attempts)

            if best_result is None:
                continue

            self.soft_kb = soft_kb
            return {
                "success": True,
                "proof": best_result.get("proof"),
                "confidence": best_result.get("confidence", 0.0),
                "soft_kb": soft_kb,
                "max_depth": depth_limit,
                "min_confidence": best_threshold,
                "attempts": attempts,
            }

        self.soft_kb = soft_kb
        return {
            "success": False,
            "proof": None,
            "confidence": 0.0,
            "soft_kb": soft_kb,
            "max_depth": self.max_depth_ceiling,
            "min_confidence": 0.0,
            "attempts": attempts,
        }
