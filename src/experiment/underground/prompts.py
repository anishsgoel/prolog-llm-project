from pathlib import Path
from typing import List

import jinja2

from solve import GoalNode
from solve.prompt_buidler import PromptBuilder, LLMSearchGuidancePromptContext


class PrologPromptBuilder(PromptBuilder):
    propose_facts: str

    _TEMPLATES_DIR = Path(__file__).parent / "templates"
    _ORDER_SCHEMA = """{
  "order": [0, 2, 1]
}"""
    _BACKTRACK_SCHEMA = """{
  "clauses": [
    {"clause": "reachable(X, Y) :- connected(X, Y).", "confidence": 0.9},
    {"clause": "connected(union_square, 14th_street).", "confidence": 0.8}
  ]
}"""

    def _make_env(self) -> jinja2.Environment:
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(self._TEMPLATES_DIR)), keep_trailing_newline=False)
        env.filters["preview_lines"] = lambda lines: "- (none)" if not lines else "\n".join(lines)
        env.filters["comment_lines"] = lambda d: "- (none)" if not d else "\n".join(f"- {p}: {c}" for p, c in d.items())
        return env

    _ENV: jinja2.Environment = None

    def __init__(self, allow_soft_rules: bool = True, max_hypotheses: int = 6, allow_new_constants: bool = False):
        self.allow_soft_rules = allow_soft_rules
        self.max_hypotheses = max_hypotheses
        self.allow_new_constants = allow_new_constants
        if PrologPromptBuilder._ENV is None:
            PrologPromptBuilder._ENV = self._make_env()

    @staticmethod
    def _candidate_lines(goal_nodes: List[GoalNode]) -> List[str]:
        return [
            f'- index={idx}; depth={node.depth}; confidence={node.confidence:.3f}; goals=[{", ".join(str(f) for f in node.unresolved_formulas())}]'
            for idx, node in enumerate(goal_nodes)
        ]

    @staticmethod
    def _clause_kind_instruction(allow_soft_rules: bool) -> str:
        return "You may propose either facts or rules." if allow_soft_rules else "You may propose facts only. Do not propose any rules."

    def order_schema(self) -> str:
        return self._ORDER_SCHEMA

    def backtrack_schema(self) -> str:
        return self._BACKTRACK_SCHEMA

    def order_prompt(self, context: LLMSearchGuidancePromptContext, goal_nodes: List[GoalNode]) -> str:
        return self._ENV.get_template("order_prompt.j2").render(context=context, candidate_lines=self._candidate_lines(goal_nodes)).strip()

    def backtrack_prompt(self, context: LLMSearchGuidancePromptContext) -> str:
        return self._ENV.get_template("backtrack_prompt.j2").render(
            context=context,
            max_hypotheses=self.max_hypotheses,
            clause_kind_instruction=self._clause_kind_instruction(self.allow_soft_rules),
            propose_facts=self.propose_facts,
            allow_new_constants=self.allow_new_constants,
        ).strip()

    def extend_on_init_prompt(self, context: LLMSearchGuidancePromptContext) -> str:
        return self._ENV.get_template("extend_on_init_prompt.j2").render(
            context=context,
            max_hypotheses=self.max_hypotheses,
            clause_kind_instruction=self._clause_kind_instruction(self.allow_soft_rules),
            propose_facts=self.propose_facts,
            allow_new_constants=self.allow_new_constants,
        ).strip()


class UndergroundPromptBuilder(PrologPromptBuilder):
    propose_facts = "connected/2"

    def __init__(self, allow_soft_rules: bool = True, max_hypotheses: int = 6, allow_new_constants: bool = False):
        super().__init__(allow_soft_rules=allow_soft_rules, max_hypotheses=max_hypotheses, allow_new_constants=allow_new_constants)


class KrebsPromptBuilder(PrologPromptBuilder):
    propose_facts = "productof/2"

    def __init__(self, allow_soft_rules: bool = True, max_hypotheses: int = 6, allow_new_constants: bool = True):
        super().__init__(allow_soft_rules=allow_soft_rules, max_hypotheses=max_hypotheses, allow_new_constants=allow_new_constants)