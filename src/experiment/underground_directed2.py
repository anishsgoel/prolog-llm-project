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
    sample_size = 5

    def preview_lines(lines) -> str:
        if not lines:
            return "- (none)"
        shown = lines[:sample_size]
        preview = list(shown)
        if len(lines) > sample_size:
            preview.append(f"- ... ({len(lines) - sample_size} more)")
        return chr(10).join(preview)

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
You are a cautious Prolog expert. You propose missing FACTS only.

GOAL:
{context.goal}

HARD KB SNAPSHOT (partial; do NOT assume any other edges exist):
{preview_lines(context.hard_fact_lines)}

Hard rules already in the program (confidence 1.0):
{preview_lines(context.hard_rule_lines)}

Predicate semantics (ground truth):
{chr(10).join(comment_lines) if comment_lines else "- (none)"}

The proof failed because these subgoals could not be proven:
{preview_lines(failed_lines)}

Task:
For each failed subgoal above, propose up to 3 additional Prolog FACTS
that are likely true and would help prove the GOAL.

Constraints:
- Output ONLY connected/2 FACTS. No rules.
- Each fact MUST end with a period.
- Prefer using only station atoms that appear in the snapshot or the GOAL.
- If a suggested edge might be a "shortcut" (skips intermediate stops), lower confidence.

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
    print(f"Proof: {result['proof']}")
    print(f"At confidence: {result['confidence']}")
    print("==============================\n")


if __name__ == "__main__":
    main()
