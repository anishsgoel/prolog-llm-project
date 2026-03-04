#!/usr/bin/env python
"""Main entry point for Prolog-LLM reasoning."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.prolog_llm import (
    KnowledgeBase,
    HardKBCollector,
    solve_with_background,
)


def main():
    config.init_config()

    kb_text = """
    1. connected(union_square, 14th_street). # connected/2 = ADJACENT STOPS ONLY. Use only when two stations are immediate neighbors on the same line. Do NOT add shortcut edges that skip intermediate stations.
    2. connected(14th_street, 23rd_street).
    3. connected(23rd_street, 34th_street).
    4. connected(34th_street, times_square).
    5. connected(times_square, 42nd_street).
    6. connected(42nd_street, grand_central).
    7. connected(grand_central, bryant_park).
    8. reachable(X, Y) :- connected(X, Y). # reachable/2 = PATH EXISTS. One-hop reachable comes only from connected/2.
    9. reachable(X, Z) :- connected(X, Y), reachable(Y, Z). # reachable/2 is the transitive closure of connected/2 (multi-hop path following only adjacent edges).
    """

    print("===== FULL METRO KB =====")
    print(kb_text)
    print("====================================\n")

    kb_obj = KnowledgeBase(kb_text)
    kb_missing_obj = kb_obj.omit_facts({7})
    kb_missing_fact = kb_missing_obj.to_text()

    print("===== METRO KB WITH FACT REMOVED =====")
    print(kb_missing_fact)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print("TEST QUERY: {}".format(test_goal))
    print("==============================\n")

    print(">>> Running HardKBCollector (hard-KB BFS)...")
    collector = HardKBCollector(kb_missing_obj, max_depth=20)
    collect_result = collector.solve(test_goal)
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=20,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)

    print("\n===== PAPER METRICS (results section) =====")
    print(bg_result.get("paper_metrics", {}))
    print("==========================================\n")

    if bg_result.get("status") == "SOFT_SUCCESS":
        print("\n===== FULL PROOF PATH (ROOT GOAL PROVEN) =====")
        for step in bg_result.get("final_proof_path", []):
            print(step)
        print("============================================\n")


if __name__ == "__main__":
    main()
