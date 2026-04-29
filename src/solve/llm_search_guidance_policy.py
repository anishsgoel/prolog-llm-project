"""LLM-backed search guidance policy for DFS proof search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import config
from prolog.formula_parsing import split_head_and_body
from prolog.knowledge_base import SoftKnowledgeBase
from prolog.prolog_command import SoftFact, SoftRule
from prolog_llm.llm import LLMInterface
from prolog_llm.prolog_utils import extract_first_json
from solve.goalnode import GoalNode
from solve.search_guidance_policy import SearchGuidancePolicy


@dataclass(frozen=True)
class LLMSearchGuidancePromptContext:
    """Context exposed to prompt builders for DFS guidance."""

    current_goal_lines: List[str]
    hard_fact_lines: List[str]
    hard_rule_lines: List[str]
    soft_fact_lines: List[str]
    soft_rule_lines: List[str]
    predicate_comments: Dict[str, str]
    min_confidence: float


class LLMSearchGuidancePolicy(SearchGuidancePolicy):
    """Guide DFS ordering and backtracking by querying an LLM."""

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        max_hypotheses: int = 6,
        allow_soft_rules: bool = True,
        order_prompt_builder: Callable[[LLMSearchGuidancePromptContext, List[GoalNode]], str] | None = None,
        backtrack_prompt_builder: Callable[[LLMSearchGuidancePromptContext], str] | None = None,
        repair_order_schema: Optional[str] = None,
        repair_backtrack_schema: Optional[str] = None,
    ):
        self.llm = llm or LLMInterface()
        self.max_hypotheses = max_hypotheses
        self.allow_soft_rules = allow_soft_rules
        self.order_prompt_builder = order_prompt_builder or self._default_order_prompt
        self.backtrack_prompt_builder = backtrack_prompt_builder or self._default_backtrack_prompt
        self.repair_order_schema = repair_order_schema or self._default_order_schema()
        self.repair_backtrack_schema = repair_backtrack_schema or self._default_backtrack_schema()

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
        goal_node: GoalNode,
    ) -> LLMSearchGuidancePromptContext:
        return LLMSearchGuidancePromptContext(
            current_goal_lines=[f"- {formula}" for formula in goal_node.unresolved_formulas()],
            hard_fact_lines=self._hard_fact_lines(soft_kb),
            hard_rule_lines=self._hard_rule_lines(soft_kb),
            soft_fact_lines=self._soft_fact_lines(soft_kb)[: config.HYP_PROMPT_FACT_LIMIT],
            soft_rule_lines=self._soft_rule_lines(soft_kb)[: config.HYP_PROMPT_FACT_LIMIT],
            predicate_comments=dict(sorted(soft_kb.predicate_comments.items())),
            min_confidence=goal_node.confidence,
        )

    def _preview_lines(self, lines: List[str]) -> str:
        if not lines:
            return "- (none)"
        return "\n".join(lines)

    def _comment_lines(self, predicate_comments: Dict[str, str]) -> str:
        if not predicate_comments:
            return "- (none)"
        return "\n".join(
            f"- {predicate}: {comment}"
            for predicate, comment in predicate_comments.items()
        )

    def _default_order_schema(self) -> str:
        return """{
  "order": [0, 2, 1]
}"""

    def _default_backtrack_schema(self) -> str:
        return """{
  "clauses": [
    {"clause": "reachable(X, Y) :- connected(X, Y).", "confidence": 0.9},
    {"clause": "connected(union_square, 14th_street).", "confidence": 0.8}
  ]
}"""

    def _default_order_prompt(
        self,
        context: LLMSearchGuidancePromptContext,
        goal_nodes: List[GoalNode],
    ) -> str:
        candidate_lines = []
        for idx, goal_node in enumerate(goal_nodes):
            formulas = ", ".join(str(formula) for formula in goal_node.unresolved_formulas())
            candidate_lines.append(
                f'- index={idx}; depth={goal_node.depth}; confidence={goal_node.confidence:.3f}; goals=[{formulas}]'
            )

        return f"""
You are a cautious Prolog proof-search guide.

Current unresolved goal formulas:
{self._preview_lines(context.current_goal_lines)}

Hard facts:
{self._preview_lines(context.hard_fact_lines)}

Hard rules:
{self._preview_lines(context.hard_rule_lines)}

Soft facts sample:
{self._preview_lines(context.soft_fact_lines)}

Soft rules sample:
{self._preview_lines(context.soft_rule_lines)}

Predicate semantics:
{self._comment_lines(context.predicate_comments)}

Candidate successor goal nodes:
{self._preview_lines(candidate_lines)}

Task:
Return the candidate indexes ordered from best to worst for DFS exploration.
Use each index at most once. Do not invent indexes.

Return ONLY valid JSON in this exact format:
{{
  "order": [0, 2, 1]
}}
""".strip()

    def _default_backtrack_prompt(
        self,
        context: LLMSearchGuidancePromptContext,
    ) -> str:
        clause_kind_instruction = (
            "You may propose either facts or rules."
            if self.allow_soft_rules
            else "You may propose facts only. Do not propose any rules."
        )

        return f"""
You are a cautious Prolog expert. You propose missing clauses that may help continue DFS proof search.

Current unresolved goal formulas:
{self._preview_lines(context.current_goal_lines)}

Hard facts:
{self._preview_lines(context.hard_fact_lines)}

Hard rules:
{self._preview_lines(context.hard_rule_lines)}

Soft facts sample:
{self._preview_lines(context.soft_fact_lines)}

Soft rules sample:
{self._preview_lines(context.soft_rule_lines)}

Predicate semantics:
{self._comment_lines(context.predicate_comments)}

Task:
The DFS search is backtracking from the current goal node.
Propose up to {self.max_hypotheses} Prolog clauses that are likely true and would help prove the current goals.
It is valid to return no clauses.

Constraints:
- {clause_kind_instruction}
- Every clause must end with a period.
- Confidence must be between 0 and 1.

Return ONLY valid JSON in this exact format:
{{
  "clauses": [
    {{"clause": "reachable(X, Y) :- connected(X, Y).", "confidence": 0.9}},
    {{"clause": "connected(oxford_circus, piccadilly_circus).", "confidence": 0.8}}
  ]
}}
""".strip()

    def order_goals(
        self,
        soft_kb: SoftKnowledgeBase,
        current_goal_node: GoalNode,
        goal_nodes: List[GoalNode],
    ) -> List[GoalNode]:
        if len(goal_nodes) <= 1:
            return goal_nodes

        context = self._prompt_context(soft_kb, current_goal_node)
        prompt = self.order_prompt_builder(context, goal_nodes)
        raw = self.llm.ask_with_retry(prompt, repair_schema=self.repair_order_schema)

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
        goal_node: GoalNode,
        soft_kb: SoftKnowledgeBase,
    ) -> Tuple[SoftKnowledgeBase, List[SoftFact], bool]:
        context = self._prompt_context(soft_kb, goal_node)
        prompt = self.backtrack_prompt_builder(context)
        raw = self.llm.ask_with_retry(prompt, repair_schema=self.repair_backtrack_schema)

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

        return new_kb, new_soft_facts,  len(new_soft_facts) != 0
