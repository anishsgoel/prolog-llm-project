"""LLM-backed extension strategy for the meta-solver."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import config
from log_setup import ERROR_LOGGER, get_logger
from logic.logic import AtomicFormula, Term, Var
from prolog.knowledge_base import SoftKnowledgeBase
from prolog.prolog_utils import is_mixed_atom
from prolog_llm.llm import LLMInterface
from prolog_llm.prolog_utils import extract_first_json
from solve.llm_search_guidance_policy import apply_clauses, coerce_confidence
from solve_bfs.extension_strategy import ExtensionStrategy

_err_log = get_logger(ERROR_LOGGER)


@dataclass(frozen=True)
class LLMPromptContext:
    """Context exposed to experiment-specific prompt builders."""

    goal: str
    hard_fact_lines: List[str]
    hard_rule_lines: List[str]
    soft_fact_lines: List[str]
    soft_rule_lines: List[str]
    predicate_comments: Dict[str, str]
    failed_atoms: List[str]
    min_confidence: float
    max_hypotheses: int
    allow_soft_rules: bool


class LLMExtensionStrategy(ExtensionStrategy):
    """Extend the soft KB by asking the LLM for missing facts and rules."""

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        max_failed_goals: int = 3,
        max_formulas_per_goal: int = 3,
        max_hypotheses: int = 6,
        increase_depth_on_empty: bool = True,
        allow_soft_rules: bool = True,
        prompt_builder: Callable[[LLMPromptContext], str] | None = None,
        repair_schema: Optional[str] = None,
    ):
        if prompt_builder is None:
            raise ValueError("LLMExtensionStrategy requires an experiment-specific prompt_builder.")

        self.llm = llm or LLMInterface()
        self.max_failed_goals = max_failed_goals
        self.max_formulas_per_goal = max_formulas_per_goal
        self.max_hypotheses = max_hypotheses
        self.increase_depth_on_empty = increase_depth_on_empty
        self.allow_soft_rules = allow_soft_rules
        self.prompt_builder = prompt_builder
        self.repair_schema = repair_schema or self._default_repair_schema()

    def _existing_clause_strings(self, soft_kb: SoftKnowledgeBase) -> Set[str]:
        clauses = {f"{fact.atom}." for fact in soft_kb.facts}
        clauses.update(f"{rule.head} :- {rule.body}." for rule in soft_kb.rules)
        return clauses

    def _standardize_atom_variables(self, formula: AtomicFormula) -> str:
        """Render an atom with short canonical variable names for prompting."""
        canonical_names = ["X", "Y", "Z", "U", "V", "W"]
        substitution: Dict[Term, Term] = {}

        for idx, variable in enumerate(sorted(formula.vars(), key=lambda current: current.name)):
            canonical_name = canonical_names[idx] if idx < len(canonical_names) else f"X{idx}"
            substitution[variable] = Var(canonical_name)

        return str(formula.substitute(substitution))

    def _failed_goal_atoms(self, failed_atoms: List[AtomicFormula]) -> List[str]:
        sampled_formulas: List[AtomicFormula] = []
        seen_signatures = set()

        for formula in failed_atoms:
            if not is_mixed_atom(formula):
                continue
            atom_signature = self._standardize_atom_variables(formula)
            if atom_signature in seen_signatures:
                continue
            seen_signatures.add(atom_signature)
            sampled_formulas.append(formula)

        limit = self.max_failed_goals * self.max_formulas_per_goal
        if len(sampled_formulas) > limit:
            sampled_formulas = random.sample(sampled_formulas, limit)

        return [self._standardize_atom_variables(formula) for formula in sampled_formulas]

    def _hard_fact_lines(self, soft_kb: SoftKnowledgeBase) -> List[str]:
        return [f"- {fact.atom}." for fact in soft_kb.facts if fact.confidence == 1.0]

    def _hard_rule_lines(self, soft_kb: SoftKnowledgeBase) -> List[str]:
        return [f"- {rule.head} :- {rule.body}." for rule in soft_kb.rules if rule.confidence == 1.0]

    def _soft_fact_lines(self, soft_kb: SoftKnowledgeBase) -> List[str]:
        return [
            f"- {fact.atom}. [conf={fact.confidence:.3f}]"
            for fact in soft_kb.facts
            if fact.confidence < 1.0
        ]

    def _soft_rule_lines(self, soft_kb: SoftKnowledgeBase) -> List[str]:
        return [
            f"- {rule.head} :- {rule.body}. [conf={rule.confidence:.3f}]"
            for rule in soft_kb.rules
            if rule.confidence < 1.0
        ]

    def _prompt_context(
        self,
        soft_kb: SoftKnowledgeBase,
        goal: AtomicFormula,
        failed_atoms: List[str],
        min_confidence: float,
    ) -> LLMPromptContext:
        return LLMPromptContext(
            goal=str(goal),
            hard_fact_lines=self._hard_fact_lines(soft_kb),
            hard_rule_lines=self._hard_rule_lines(soft_kb),
            soft_fact_lines=self._soft_fact_lines(soft_kb)[: config.HYP_PROMPT_FACT_LIMIT],
            soft_rule_lines=self._soft_rule_lines(soft_kb)[: config.HYP_PROMPT_FACT_LIMIT],
            predicate_comments=dict(sorted(soft_kb.predicate_comments.items())),
            failed_atoms=failed_atoms,
            min_confidence=min_confidence,
            max_hypotheses=self.max_hypotheses,
            allow_soft_rules=self.allow_soft_rules,
        )

    def _default_repair_schema(self) -> str:
        return """{
  "clauses": [
    {"clause": "reachable(X, Y) :- connected(X, Y).", "confidence": 0.9},
    {"clause": "connected(union_square, 14th_street).", "confidence": 0.8}
  ]
}"""

    def extend(
        self,
        soft_kb: SoftKnowledgeBase,
        goal: AtomicFormula,
        failed_atoms: List[AtomicFormula],
        max_depth: int,
        min_confidence: float,
    ) -> Tuple[SoftKnowledgeBase, int, float]:
        failed_atom_strings = self._failed_goal_atoms(failed_atoms)
        if not failed_atom_strings:
            next_depth = max_depth + 1 if self.increase_depth_on_empty else max_depth
            return soft_kb, next_depth, min_confidence

        prompt = self.prompt_builder(self._prompt_context(soft_kb, goal, failed_atom_strings, min_confidence))
        raw = self.llm.ask_with_retry(prompt, repair_schema=self.repair_schema)

        try:
            data = json.loads(extract_first_json(raw))
        except Exception:
            _err_log.exception("LLMExtensionStrategy.extend: failed to parse LLM clauses response. raw=%r", raw)
            print("Extension empty.")
            next_depth = max_depth + 1 if self.increase_depth_on_empty else max_depth
            return soft_kb, next_depth, min_confidence

        clauses = data.get("clauses", [])
        if not isinstance(clauses, list):
            print("no clauses")
            next_depth = max_depth + 1 if self.increase_depth_on_empty else max_depth
            return soft_kb, next_depth, min_confidence

        items = clauses[: self.max_hypotheses]

        # Highest confidence the LLM OFFERED (before insertion), including
        # proposals that turn out to be duplicates -- this drives the depth /
        # confidence-floor decision below, so it must not be narrowed to the
        # proposals that happened to land.
        offered = [c for c in (coerce_confidence(item) for item in items) if c is not None]
        max_confidence_added = max([0.0] + offered)

        # Insertion is the SHARED helper, not a second copy of it. The previous
        # inline loop here differed from apply_clauses in two ways, both bugs:
        # it set `added = True` unconditionally (so a batch of rejected duplicates
        # still reported an extension), and it advanced `next_num` whenever
        # add_soft_fact returned True -- including on a confidence raise, where
        # nothing was appended, which drifted the clause numbering.
        new_kb, _new_soft_facts, facts_added = apply_clauses(
            items, soft_kb, allow_soft_rules=self.allow_soft_rules
        )
        # apply_clauses' third value covers facts only; rules count as an
        # extension here too (see its docstring).
        added = facts_added or len(new_kb.rules) > len(soft_kb.rules)

        if not added:
            print("not added!")
            next_depth = max_depth + 1 if self.increase_depth_on_empty else max_depth
            return soft_kb, next_depth, min_confidence

        if max_confidence_added < min_confidence:
            max_depth += 1
        return new_kb, max_depth, max(min_confidence * 0.9, min(min_confidence, max_confidence_added))
