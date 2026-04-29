from typing import List, Dict

from solve import GoalNode
from solve.llm_search_guidance import PromptBuilder, LLMSearchGuidancePromptContext


class UndergroundPromptBuilder(PromptBuilder):
    def __init__(self, allow_soft_rules: bool = True, max_hypotheses: int = 6):
        self.allow_soft_rules = allow_soft_rules
        self.max_hypotheses = max_hypotheses

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

    def order_schema(self) -> str:
        return """{
  "order": [0, 2, 1]
}"""

    def backtrack_schema(self) -> str:
        return """{
  "clauses": [
    {"clause": "reachable(X, Y) :- connected(X, Y).", "confidence": 0.9},
    {"clause": "connected(union_square, 14th_street).", "confidence": 0.8}
  ]
}"""

    def order_prompt(
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

The goal of the program is to prove the following query:
{context.goal}

Current state of the search:
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

    def backtrack_prompt(
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

The goal of the program is to prove the following query:
{context.goal}

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
- Output ONLY connected/2 FACTS. No rules.
- {clause_kind_instruction}
- Every clause must end with a period.
- Confidence must be between 0 and 1.
- Prefer using only station atoms that appear in the snapshot or the GOAL.
- If a suggested edge might be a "shortcut" (skips intermediate stops), lower confidence.
- Propose only clauses that are likely true, if you believe there is no formula that is missing in the current program,
    return empty list.

Return ONLY valid JSON in this exact format:
{{
  "clauses": [
    {{"clause": "reachable(X, Y) :- connected(X, Y).", "confidence": 0.9}},
    {{"clause": "connected(oxford_circus, piccadilly_circus).", "confidence": 0.8}}
  ]
}}
""".strip()
