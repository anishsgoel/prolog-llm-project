#!/usr/bin/env python
"""Main entry point for Prolog-LLM reasoning."""

import sys
import os

from prolog.formula_parsing import parse_prolog_to_formula
from solve import MetaSolver, LLMExtensionStrategy
from solve.solver import Solver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from prolog.knowledge_base import KnowledgeBase


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
    test_goal_formula = parse_prolog_to_formula(test_goal)

    print("==============================")
    print("TEST QUERY: {}".format(test_goal))
    print("==============================\n")

    print("==============================")
    print("PREDICATE COMMENTS: {}".format(kb_obj.predicate_comments))
    print("==============================\n")

    print("==============================")
    print("solving")
    s = Solver(kb_missing_obj, 10)
    s = Solver(kb_obj, 10)
    result = s.solve(test_goal_formula)
    print(f"Result solved {result['success']}")
    print("==============================\n")

    print("==============================")
    print("solving meta")
    s = MetaSolver(kb_missing_obj, LLMExtensionStrategy(allow_soft_rules=False), max_depth=5, min_confidence=1.0, max_rounds=10)
    result = s.solve(test_goal_formula)
    print(f"Result solved {result['success']}")
    print("==============================\n")




if __name__ == "__main__":
    main()
