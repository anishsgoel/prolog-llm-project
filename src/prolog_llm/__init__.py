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

from prolog_llm.knowledge_base import (
    KnowledgeBase,
    Fact,
    Rule,
)

from prolog_llm.soft_kb import (
    SoftKB,
    SoftFact,
    SoftRule,
)

from prolog_llm.solvers import (
    PrologSolver,
    HardKBCollector,
    SoftBFSSolver,
)

from prolog_llm.hypothesis_generator import (
    HypothesisGenerator,
    generate_background_hypotheses_fast,
    attach_hypotheses_to_kb,
)

from prolog_llm.llm_interface import (
    LLMInterface,
    get_llm_interface,
    ask_llm,
    llm_json_only,
    nl_to_prolog_kb,
)

from prolog_llm.orchestration import (
    solve_with_background,
)

__all__ = [
    # Utils
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
    # Knowledge Base
    "KnowledgeBase",
    "Fact",
    "Rule",
    # Soft KB
    "SoftKB",
    "SoftFact",
    "SoftRule",
    # Solvers
    "PrologSolver",
    "HardKBCollector",
    "SoftBFSSolver",
    # Hypothesis Generator
    "HypothesisGenerator",
    "generate_background_hypotheses_fast",
    "attach_hypotheses_to_kb",
    # LLM
    "LLMInterface",
    "get_llm_interface",
    "ask_llm",
    "llm_json_only",
    "nl_to_prolog_kb",
    # Orchestration
    "solve_with_background",
]
