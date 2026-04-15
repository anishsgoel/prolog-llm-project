#!/usr/bin/env python
"""Experiment entry point for the undirected underground task."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from prolog.formula_parsing import parse_prolog_to_formula
from prolog.knowledge_base import KnowledgeBase
from solve import LLMExtensionStrategy, MetaSolver
from solve.llm_extension_strategy import LLMPromptContext
from solve.solver import Solver


def build_prompt(context: LLMPromptContext) -> str:
    """Build the experiment-specific LLM prompt."""
    comment_lines = [
        f"- {predicate}: {comment}"
        for predicate, comment in context.predicate_comments.items()
    ]
    failed_lines = [f"- {atom}" for atom in context.failed_atoms]
    clause_kind_instruction = (
        "You may propose either facts or rules."
        if context.allow_soft_rules
        else "You may propose facts only. Do not propose any rules."
    )

    return f"""
You are extending a soft Prolog knowledge base to help prove currently failing goals.

Hard facts already in the program (confidence 1.0):
{chr(10).join(context.hard_fact_lines) if context.hard_fact_lines else "- (none)"}

Hard rules already in the program (confidence 1.0):
{chr(10).join(context.hard_rule_lines) if context.hard_rule_lines else "- (none)"}

Existing soft facts:
{chr(10).join(context.soft_fact_lines) if context.soft_fact_lines else "- (none)"}

Existing soft rules:
{chr(10).join(context.soft_rule_lines) if context.soft_rule_lines else "- (none)"}

Predicate comments:
{chr(10).join(comment_lines) if comment_lines else "- (none)"}

Failed unresolved atoms:
{chr(10).join(failed_lines) if failed_lines else "- (none)"}

Task:
- Propose up to {context.max_hypotheses} new Prolog clauses that could help prove the failed atoms.
- {clause_kind_instruction}
- Keep predicate names consistent with the existing KB.
- Use confidence values between 0.0 and 1.0.
- Return only clauses that are syntactically valid Prolog.
- Do not repeat clauses that already appear in the KB.
- Try to answer with keeping in mind the real world meaning of the constants and try to avoid
  false positives as possible.

Return ONLY valid JSON in this exact format:
{{
  "clauses": [
    {{"clause": "reachable(X, Y) :- connected(X, Y).", "confidence": 0.9}},
    {{"clause": "connected(14th_street, 23rd_street).", "confidence": 0.8}}
  ]
}}
""".strip()


def main() -> None:
    config.init_config()

    kb_text = """
    1. connected(union_square, 14th_street). # connected/2 = ADJACENT STOPS ONLY. Use only when two stations are immediate neighbors on the same line. Do NOT add shortcut edges that skip intermediate stations.  The stations are connected in New York underground.
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
    s = MetaSolver(
        kb_missing_obj,
        LLMExtensionStrategy(
            allow_soft_rules=True,
            prompt_builder=build_prompt,
        ),
        max_depth=5,
        min_confidence=1.0,
        max_rounds=10,
    )
    result = s.solve(test_goal_formula)
    print(f"Result solved {result['success']}")
    print("==============================\n")


if __name__ == "__main__":
    main()
