"""LLM-backed search guidance policy for DFS proof search."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import config
from logic.logic import AtomicFormula
from prolog.formula_parsing import split_head_and_body
from prolog.knowledge_base import SoftKnowledgeBase
from prolog.prolog_command import SoftFact, SoftRule
from prolog_llm.llm import LLMInterface
from prolog_llm.prolog_utils import extract_first_json
from solve.goalnode import GoalNode
from solve.llm_search_guidance import PromptBuilder, LLMSearchGuidancePromptContext
from solve.search_guidance_policy import SearchGuidancePolicy


class LLMSearchGuidancePolicy(SearchGuidancePolicy):
    """Guide DFS ordering and backtracking by querying an LLM."""

    def __init__(
            self,
            llm_search_guidance: PromptBuilder,
            llm: Optional[LLMInterface] = None,
            max_hypotheses: int = 6,
            allow_soft_rules: bool = True,
    ):
        self.llm = llm or LLMInterface()
        self.max_hypotheses = max_hypotheses
        self.allow_soft_rules = allow_soft_rules
        self.llm_search_guidance = llm_search_guidance

    def _hard_fact_lines(self, soft_kb: SoftKnowledgeBase) -> List[str]:
        return [f"- {fact.atom}." for fact in soft_kb.facts if fact.confidence == 1.0]

    def _hard_rule_lines(self, soft_kb: SoftKnowledgeBase) -> List[str]:
        return [f"- {rule.head} :- {rule.body}." for rule in soft_kb.rules if rule.confidence == 1.0]

    def _soft_fact_lines(self, soft_kb: SoftKnowledgeBase, min_confidence: float) -> List[str]:
        return [
            f"- {fact.atom}. [conf={fact.confidence:.3f}]"
            for fact in soft_kb.facts
            if min_confidence <= fact.confidence < 1.0
        ]

    def _soft_rule_lines(self, soft_kb: SoftKnowledgeBase, min_confidence: float) -> List[str]:
        return [
            f"- {rule.head} :- {rule.body}. [conf={rule.confidence:.3f}]"
            for rule in soft_kb.rules
            if min_confidence <= rule.confidence < 1.0
        ]

    def _prompt_context(
            self,
            goal: AtomicFormula,
            soft_kb: SoftKnowledgeBase,
            goal_node: GoalNode,
            min_confidence: float,
    ) -> LLMSearchGuidancePromptContext:
        return LLMSearchGuidancePromptContext(
            goal=str(goal),
            current_goal_lines=[f"- {formula}" for formula in goal_node.unresolved_formulas()],
            hard_fact_lines=self._hard_fact_lines(soft_kb),
            hard_rule_lines=self._hard_rule_lines(soft_kb),
            soft_fact_lines=self._soft_fact_lines(soft_kb, min_confidence)[: config.HYP_PROMPT_FACT_LIMIT],
            soft_rule_lines=self._soft_rule_lines(soft_kb, min_confidence)[: config.HYP_PROMPT_FACT_LIMIT],
            predicate_comments=dict(sorted(soft_kb.predicate_comments.items())),
            min_confidence=min_confidence,
        )

    def order_goals(
            self,
            goal: AtomicFormula,
            soft_kb: SoftKnowledgeBase,
            current_goal_node: GoalNode,
            min_confidence: float,
            goal_nodes: List[GoalNode],
    ) -> List[GoalNode]:
        if len(goal_nodes) <= 1:
            return goal_nodes

        context = self._prompt_context(goal, soft_kb, current_goal_node, min_confidence)
        prompt = self.llm_search_guidance.order_prompt(context, goal_nodes)
        raw = self.llm.ask_with_retry(prompt, repair_schema=self.llm_search_guidance.order_schema())

        try:
            data = json.loads(extract_first_json(raw))
            order = data.get("order", [])
        except Exception:
            return goal_nodes

        if not isinstance(order, list):
            return goal_nodes

        ordered_goal_nodes: List[GoalNode] = []
        used_indexes = set()

        for idx in order:
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= len(goal_nodes):
                continue
            if idx in used_indexes:
                continue
            ordered_goal_nodes.append(goal_nodes[idx])
            used_indexes.add(idx)

        for idx, goal_node in enumerate(goal_nodes):
            if idx not in used_indexes:
                ordered_goal_nodes.append(goal_node)

        return ordered_goal_nodes

    def extend_on_backtrack(
            self,
            goal: AtomicFormula,
            goal_node: GoalNode,
            min_confidence: float,
            soft_kb: SoftKnowledgeBase,

    ) -> Tuple[SoftKnowledgeBase, List[SoftFact], bool]:
        context = self._prompt_context(goal, soft_kb, goal_node, min_confidence)
        prompt = self.llm_search_guidance.backtrack_prompt(context)
        raw = self.llm.ask_with_retry(prompt, repair_schema=self.llm_search_guidance.backtrack_schema())

        try:
            data = json.loads(extract_first_json(raw))
        except Exception:
            return soft_kb, [], False

        clauses = data.get("clauses", [])
        if not isinstance(clauses, list):
            return soft_kb, [], False

        new_kb = soft_kb.copy()
        next_num = max([0] + [fact.num for fact in new_kb.facts] + [rule.num for rule in new_kb.rules]) + 1
        new_soft_facts: List[SoftFact] = []

        for item in clauses:
            if not isinstance(item, dict):
                continue

            try:
                confidence = float(item.get("confidence", 0.0))
            except Exception:
                continue
            confidence = max(0.0, min(1.0, confidence))

            head_or_atom, maybe_body = split_head_and_body(item.get("clause", ""))
            if head_or_atom is None and maybe_body is None:
                continue

            if maybe_body is None:
                before_fact_count = len(new_kb.facts)
                changed = new_kb.add_soft_fact(next_num, head_or_atom, confidence)
                if changed:
                    matching_fact = None
                    for fact in new_kb.facts:
                        if fact.atom == head_or_atom and abs(fact.confidence - confidence) < 1e-9:
                            matching_fact = fact
                            break
                    if matching_fact is None and len(new_kb.facts) > before_fact_count:
                        matching_fact = new_kb.facts[-1]
                    if matching_fact is not None:
                        new_soft_facts.append(matching_fact)
                    if len(new_kb.facts) > before_fact_count:
                        next_num += 1
                continue

            if not self.allow_soft_rules:
                continue

            new_kb.rules.append(SoftRule(next_num, head_or_atom, maybe_body, confidence))
            next_num += 1

        return new_kb, new_soft_facts, len(new_soft_facts) != 0
