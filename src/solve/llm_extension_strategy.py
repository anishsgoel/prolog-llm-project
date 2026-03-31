"""LLM-backed extension strategy for the meta-solver."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Set, Tuple

import config
from prolog.formula_parsing import parse_predicate, split_body_atoms
from prolog.knowledge_base import SoftKnowledgeBase
from prolog.prolog_command import SoftFact, SoftRule
from prolog_llm.llm import LLMInterface
from prolog_llm.prolog_utils import extract_first_json
from solve.extension_strategy import ExtensionStrategy
from solve.goalnode import GoalNode


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
    ):
        self.llm = llm or LLMInterface()
        self.max_failed_goals = max_failed_goals
        self.max_formulas_per_goal = max_formulas_per_goal
        self.max_hypotheses = max_hypotheses
        self.increase_depth_on_empty = increase_depth_on_empty
        self.allow_soft_rules = allow_soft_rules

    def _existing_clause_strings(self, soft_kb: SoftKnowledgeBase) -> Set[str]:
        clauses = {f"{fact.atom}." for fact in soft_kb.facts}
        clauses.update(f"{rule.head} :- {rule.body}." for rule in soft_kb.rules)
        return clauses

    def _failed_goal_atoms(self, failed_goals: List[GoalNode]) -> List[str]:
        atoms: List[str] = []
        seen = set()
        for goal in failed_goals[:self.max_failed_goals]:
            for formula in goal.unresolved_formulas()[:self.max_formulas_per_goal]:
                atom = str(formula)
                if atom not in seen:
                    seen.add(atom)
                    atoms.append(atom)
        return atoms

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

    def _prompt(self, soft_kb: SoftKnowledgeBase, failed_atoms: List[str], min_confidence: float) -> str:
        hard_fact_lines = self._hard_fact_lines(soft_kb)
        hard_rule_lines = self._hard_rule_lines(soft_kb)
        soft_fact_lines = self._soft_fact_lines(soft_kb)[: config.HYP_PROMPT_FACT_LIMIT]
        soft_rule_lines = self._soft_rule_lines(soft_kb)[: config.HYP_PROMPT_FACT_LIMIT]
        comment_lines = [
            f"- {predicate}: {comment}"
            for predicate, comment in sorted(soft_kb.predicate_comments.items())
        ]
        failed_lines = [f"- {atom}" for atom in failed_atoms]
        clause_kind_instruction = (
            "You may propose either facts or rules."
            if self.allow_soft_rules
            else "You may propose facts only. Do not propose any rules."
        )

        return f"""
You are extending a soft Prolog knowledge base to help prove currently failing goals.

Hard facts already in the program (confidence 1.0):
{chr(10).join(hard_fact_lines) if hard_fact_lines else "- (none)"}

Hard rules already in the program (confidence 1.0):
{chr(10).join(hard_rule_lines) if hard_rule_lines else "- (none)"}

Existing soft facts:
{chr(10).join(soft_fact_lines) if soft_fact_lines else "- (none)"}

Existing soft rules:
{chr(10).join(soft_rule_lines) if soft_rule_lines else "- (none)"}

Predicate comments:
{chr(10).join(comment_lines) if comment_lines else "- (none)"}

Failed unresolved atoms:
{chr(10).join(failed_lines) if failed_lines else "- (none)"}

Task:
- Propose up to {self.max_hypotheses} new Prolog clauses that could help prove the failed atoms.
- {clause_kind_instruction}
- Keep predicate names consistent with the existing KB.
- Use confidence values between {min_confidence:.2f} and 1.0.
- Return only clauses that are syntactically valid Prolog.
- Do not repeat clauses that already appear in the KB.

Return ONLY valid JSON in this exact format:
{{
  "clauses": [
    {{"clause": "reachable(X, Y) :- connected(X, Y).", "confidence": 0.9}},
    {{"clause": "connected(grand_central, bryant_park).", "confidence": 0.8}}
  ]
}}
""".strip()

    def _repair_schema(self) -> str:
        return """{
  "clauses": [
    {"clause": "reachable(X, Y) :- connected(X, Y).", "confidence": 0.9},
    {"clause": "connected(grand_central, bryant_park).", "confidence": 0.8}
  ]
}"""

    def _normalize_clause(self, clause: str) -> Optional[Tuple[str, Optional[str]]]:
        clause = (clause or "").strip()
        if not clause:
            return None
        if not clause.endswith("."):
            clause += "."

        body = clause[:-1].strip()
        if ":-" in body:
            head, body_part = body.split(":-", 1)
            head = head.strip()
            body_part = body_part.strip()
            try:
                parse_predicate(head)
                for atom in split_body_atoms(body_part):
                    parse_predicate(atom.strip())
            except Exception:
                return None
            return head, body_part

        try:
            parse_predicate(body)
        except Exception:
            return None
        return body, None

    def extend(
        self,
        soft_kb: SoftKnowledgeBase,
        failed_goals: List[GoalNode],
        max_depth: int,
        min_confidence: float,
    ) -> Tuple[SoftKnowledgeBase, int, float]:
        failed_atoms = self._failed_goal_atoms(failed_goals)
        if not failed_atoms:
            next_depth = max_depth + 1 if self.increase_depth_on_empty else max_depth
            return soft_kb, next_depth, min_confidence

        prompt = self._prompt(soft_kb, failed_atoms, min_confidence)
        raw = self.llm.ask_with_retry(prompt, repair_schema=self._repair_schema())

        try:
            data = json.loads(extract_first_json(raw))
        except Exception:
            next_depth = max_depth + 1 if self.increase_depth_on_empty else max_depth
            return soft_kb, next_depth, min_confidence

        clauses = data.get("clauses", [])
        if not isinstance(clauses, list):
            next_depth = max_depth + 1 if self.increase_depth_on_empty else max_depth
            return soft_kb, next_depth, min_confidence

        existing = self._existing_clause_strings(soft_kb)
        new_kb = soft_kb.copy()
        next_num = max([0] + [fact.num for fact in new_kb.facts] + [rule.num for rule in new_kb.rules]) + 1
        added = False

        for item in clauses[: self.max_hypotheses]:
            if not isinstance(item, dict):
                continue

            try:
                confidence = float(item.get("confidence", 0.0))
            except Exception:
                continue
            confidence = max(0.0, min(1.0, confidence))
            if confidence < min_confidence:
                continue

            normalized = self._normalize_clause(item.get("clause", ""))
            if normalized is None:
                continue

            head_or_atom, maybe_body = normalized
            if maybe_body is not None and not self.allow_soft_rules:
                continue
            clause_text = f"{head_or_atom}." if maybe_body is None else f"{head_or_atom} :- {maybe_body}."
            if clause_text in existing:
                continue

            if maybe_body is None:
                new_kb.facts.append(SoftFact(next_num, head_or_atom, confidence))
            else:
                new_kb.rules.append(SoftRule(next_num, head_or_atom, maybe_body, confidence))

            existing.add(clause_text)
            next_num += 1
            added = True

        if not added:
            next_depth = max_depth + 1 if self.increase_depth_on_empty else max_depth
            return soft_kb, next_depth, min_confidence

        return new_kb, max_depth, min_confidence
