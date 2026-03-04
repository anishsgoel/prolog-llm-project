"""Main orchestration functions for Prolog-LLM reasoning."""

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prolog_llm.solvers import (
    HardKBCollector,
    SoftBFSSolver,
)
from prolog_llm.hypothesis_generator import HypothesisGenerator
from prolog_llm.knowledge_base import KnowledgeBase
from prolog_llm.soft_kb import SoftKB
import config


def solve_with_background(
    goal: str,
    kb: str,
    max_depth: Optional[int] = None,
    max_soft=None,
    hard_result=None,
):
    """Main orchestration: solve with background hypothesis generation."""
    if max_depth is None:
        max_depth = config.DEFAULT_MAX_DEPTH

    kb_obj = KnowledgeBase(kb)
    predicate_comments = kb_obj.predicate_comments

    if config.VERBOSE:
        print("\n========================================")
        print(f"SOLVE WITH BACKGROUND: {goal}")
        print("========================================\n")

    if hard_result is None:
        if config.VERBOSE:
            print(">>> Phase 1: Hard-KB BFS (HardKBCollector)")
        collector = HardKBCollector(kb_obj, max_depth=max_depth)
        hard_result = collector.solve(goal)
        if config.VERBOSE:
            print("Hard-KB result:", hard_result)
    else:
        if config.VERBOSE:
            print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
            print("Hard-KB result:", hard_result)

    if hard_result.get("success"):
        if config.VERBOSE:
            print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": [],
            "final_proof_path": hard_result.get("proof_path", []),
            "paper_metrics": {
                "hard_expansions": hard_result.get("metrics", {}).get("hard_expansions"),
                "blocking_atom": None,
                "llm_hypotheses_considered": 0,
                "soft_cost": 0,
                "min_conf": None,
            }
        }

    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        if config.VERBOSE:
            print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
            print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": [],
            "final_proof_path": [],
            "paper_metrics": {
                "hard_expansions": hard_result.get("metrics", {}).get("hard_expansions"),
                "blocking_atom": None,
                "llm_hypotheses_considered": 0,
                "soft_cost": None,
                "min_conf": None,
            }
        }

    if config.VERBOSE:
        print("\n>>> Phase 2: Generate background hypotheses (HypothesisGenerator)")
        print("Unresolved atoms (minimal):", unresolved_atoms)

    hyp_gen = HypothesisGenerator()
    hypotheses = hyp_gen.generate(goal, kb_obj, hard_result)

    if not hypotheses:
        if config.VERBOSE:
            print("Hypotheses returned by LLM: []")
            print("\nLLM returned NO hypotheses; cannot build soft KB.")
            print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": [],
            "final_proof_path": [],
            "paper_metrics": {
                "hard_expansions": hard_result.get("metrics", {}).get("hard_expansions"),
                "blocking_atom": next(iter(unresolved_atoms)) if unresolved_atoms else None,
                "llm_hypotheses_considered": 0,
                "soft_cost": None,
                "min_conf": None,
            }
        }

    if config.VERBOSE:
        print("Hypotheses returned by LLM:")
        for h in hypotheses:
            print("  - Clause:", h.get("clause"),
                  "| Conf:", h.get("confidence"),
                  "| From atom:", h.get("from_atom"),
                  "| Shortcut:", h.get("is_shortcut"),
                  "| UnknownStation:", h.get("unknown_station"))

        print("\n>>> Phase 3: Attach hypotheses to soft KB")

    soft_kb = SoftKB.from_hypotheses(hypotheses, kb)

    if config.VERBOSE:
        print("Soft KB facts:", soft_kb.facts)
        print("Soft KB rules:", soft_kb.rules)
        print("\n>>> Phase 4: Soft BFS (SoftBFSSolver)")

    soft_solver = SoftBFSSolver(kb_obj, soft_kb, max_depth=max_depth, max_soft=max_soft)
    soft_result = soft_solver.solve(goal)

    if config.VERBOSE:
        print("Soft-BFS result:", soft_result)

    if soft_result.get("success"):
        if config.VERBOSE:
            print("\n>>> Result: SOFT_SUCCESS (proof found using background hypotheses)\n")
        final_path = soft_result.get("proof_path", [])
        return {
            "status": "SOFT_SUCCESS",
            "hard_result": hard_result,
            "soft_result": soft_result,
            "hypotheses": hypotheses,
            "final_proof_path": final_path,
            "paper_metrics": {
                "hard_expansions": hard_result.get("metrics", {}).get("hard_expansions"),
                "blocking_atom": hard_result.get("metrics", {}).get("blocking_atom") or (next(iter(unresolved_atoms)) if unresolved_atoms else None),
                "llm_hypotheses_considered": len(hypotheses),
                "llm_hypotheses_injected": len(soft_kb.facts) + len(soft_kb.rules),
                "soft_cost": soft_result.get("soft_cost"),
                "min_conf": soft_result.get("min_conf"),
                "penalty_sum": soft_result.get("penalty_sum"),
                "depth": soft_result.get("depth"),
            }
        }

    if config.VERBOSE:
        print("\n>>> Result: SOFT_FAILURE (no proof even with background hypotheses)\n")

    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "soft_result": soft_result,
        "hypotheses": hypotheses,
        "final_proof_path": [],
        "paper_metrics": {
            "hard_expansions": hard_result.get("metrics", {}).get("hard_expansions"),
            "blocking_atom": hard_result.get("metrics", {}).get("blocking_atom") or (next(iter(unresolved_atoms)) if unresolved_atoms else None),
            "llm_hypotheses_considered": len(hypotheses),
            "llm_hypotheses_injected": len(soft_kb.facts) + len(soft_kb.rules),
            "soft_cost": None,
            "min_conf": None,
        }
    }
