"""Meta-solver orchestration over repeated solver runs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import config
from logic.logic import AtomicFormula
from prolog.knowledge_base import KnowledgeBase, SoftKnowledgeBase
from solve.extension_strategy import ExtensionStrategy
from solve.solver import Solver


class MetaSolver:
    """Repeatedly run the solver and extend its limits until proof or termination."""

    def __init__(
        self,
        kb: KnowledgeBase,
        extension_strategy: ExtensionStrategy,
        max_depth: Optional[int] = None,
        min_confidence: float = 0.0,
        max_rounds: int = 10,
    ):
        self.hard_kb = kb
        self.soft_kb = SoftKnowledgeBase(kb)
        self.extension_strategy = extension_strategy
        self.max_depth = max_depth or config.DEFAULT_MAX_DEPTH
        self.min_confidence = min_confidence
        self.max_rounds = max_rounds

    def _log_extension(
        self,
        old_soft_kb: SoftKnowledgeBase,
        new_soft_kb: SoftKnowledgeBase,
        old_max_depth: int,
        new_max_depth: int,
        old_min_confidence: float,
        new_min_confidence: float,
    ) -> None:
        """Log newly added clauses and updated solver parameters."""
        if not config.VERBOSE:
            return

        print("[MetaSolver] Extension step")
        print(
            "[MetaSolver] Parameters: max_depth {} -> {}, min_confidence {:.3f} -> {:.3f}".format(
                old_max_depth,
                new_max_depth,
                old_min_confidence,
                new_min_confidence,
            )
        )

        old_fact_nums = {fact.num for fact in old_soft_kb.facts}
        old_rule_nums = {rule.num for rule in old_soft_kb.rules}

        new_facts = [fact for fact in new_soft_kb.facts if fact.num not in old_fact_nums]
        new_rules = [rule for rule in new_soft_kb.rules if rule.num not in old_rule_nums]

        if new_facts:
            print("[MetaSolver] New facts:")
            for fact in new_facts:
                print(f"  - {fact}")

        if new_rules:
            print("[MetaSolver] New rules:")
            for rule in new_rules:
                print(f"  - {rule}")

        if not new_facts and not new_rules:
            print("[MetaSolver] No new clauses added.")

    def solve(self, goal: AtomicFormula) -> Dict[str, Any]:
        """Run iterative solving with extension steps between failed attempts."""
        soft_kb = self.soft_kb
        max_depth = self.max_depth
        min_confidence = self.min_confidence

        attempts: List[Dict[str, Any]] = []

        for _ in range(self.max_rounds):
            solver = Solver(soft_kb, max_depth=max_depth)
            result = solver.solve(goal, min_confidence=min_confidence)
            attempts.append(result)

            if result.get("success"):
                return {
                    "success": True,
                    "proof": result.get("proof"),
                    "confidence": result.get("confidence", 0.0),
                    "failed_atoms": result.get("failed_atoms", []),
                    "soft_kb": soft_kb,
                    "max_depth": max_depth,
                    "min_confidence": min_confidence,
                    "attempts": attempts,
                }

            next_soft_kb, next_max_depth, next_min_confidence = self.extension_strategy.extend(
                soft_kb=soft_kb,
                goal=goal,
                failed_atoms=result.get("failed_atoms", []),
                max_depth=max_depth,
                min_confidence=min_confidence,
            )

            if (
                next_soft_kb is soft_kb
                and next_max_depth == max_depth
                and next_min_confidence == min_confidence
            ):
                break

            self._log_extension(
                old_soft_kb=soft_kb,
                new_soft_kb=next_soft_kb,
                old_max_depth=max_depth,
                new_max_depth=next_max_depth,
                old_min_confidence=min_confidence,
                new_min_confidence=next_min_confidence,
            )

            soft_kb = next_soft_kb
            max_depth = next_max_depth
            min_confidence = next_min_confidence

        return {
            "success": False,
            "proof": None,
            "confidence": 0.0,
            "failed_atoms": attempts[-1].get("failed_atoms", []) if attempts else [],
            "soft_kb": soft_kb,
            "max_depth": max_depth,
            "min_confidence": min_confidence,
            "attempts": attempts,
        }
