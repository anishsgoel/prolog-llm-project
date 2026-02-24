"""Prolog-LLM Reasoning Package.

A system for solving Prolog goals using LLM-assisted background hypothesis generation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prolog_llm.prolog_utils import (
    parse_predicate,
    is_variable,
    split_inline_comment,
    strip_inline_comment,
    check_exact_match,
    unify_with_fact,
    apply_bindings,
    find_matching_rules_only,
    get_subgoals,
    is_ground_atom,
    extract_first_json,
)

from prolog_llm.solvers import (
    bfs_prolog_collect,
    bfs_prolog_metro_soft,
)

from prolog_llm.hypotheses import (
    generate_background_hypotheses_fast,
    attach_hypotheses_to_kb,
)

from prolog_llm.llm_interface import (
    ask_llm,
    llm_json_only,
    nl_to_prolog_kb,
)

from prolog_llm.orchestration import (
    solve_with_background,
    omit_facts_from_kb,
    parse_kb_predicate_comments,
)

__all__ = [
    "parse_predicate",
    "is_variable",
    "split_inline_comment",
    "strip_inline_comment",
    "check_exact_match",
    "unify_with_fact",
    "apply_bindings",
    "find_matching_rules_only",
    "get_subgoals",
    "is_ground_atom",
    "extract_first_json",
    "bfs_prolog_collect",
    "bfs_prolog_metro_soft",
    "generate_background_hypotheses_fast",
    "attach_hypotheses_to_kb",
    "ask_llm",
    "llm_json_only",
    "nl_to_prolog_kb",
    "solve_with_background",
    "omit_facts_from_kb",
    "parse_kb_predicate_comments",
]
