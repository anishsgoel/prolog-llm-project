"""Prolog command representations (facts and rules)."""

from typing import Optional, List, Tuple

from logic.logic import Term, AtomicFormula
from prolog.formula_parsing import (
    parse_prolog_to_formula,
    parse_body_to_formula, parse_predicate,
)


class Fact:
    """A Prolog fact."""
    
    def __init__(self, num: int, atom: str | AtomicFormula):
        self.num = num
        if isinstance(atom, str):
            self._atom_str = atom
            self.formula = parse_prolog_to_formula(atom)
        elif isinstance(atom, AtomicFormula):
            self._atom_str = str(atom)
            self.formula = parse_prolog_to_formula(self._atom_str)
        else:
            raise TypeError("Atom can be only atomic formula or string.")

    @property
    def atom(self) -> str:
        return self._atom_str

    @property
    def predicate_name(self) -> str:
        return self.formula.pred.name

    @property
    def arity(self) -> int:
        return self.formula.pred.arity

    @property
    def args(self) -> Tuple[Term]:
        return self.formula.args

    @property
    def confidence(self) -> float:
        return 1.0
    
    def __repr__(self) -> str:
        return "Fact({}, {})".format(self.num, self._atom_str)


class Rule:
    """A Prolog rule."""
    
    def __init__(self, num: int, head: str, body: str):
        self.num = num
        self._head_str = head
        self._body_str = body
        self.head_formula = parse_prolog_to_formula(head)
        self.body_formula = parse_body_to_formula(body)

    @property
    def head(self) -> str:
        return self._head_str
    
    @property
    def body(self) -> str:
        return self._body_str

    @property
    def confidence(self) -> float:
        return 1.0
    
    def __repr__(self) -> str:
        return "Rule({}, {} :- {})".format(self.num, self._head_str, self._body_str)


class SoftFact(Fact):
    """A soft (hypothesized) fact with confidence."""

    def __init__(self, num: int, atom: str | AtomicFormula, confidence: float):
        super().__init__(num, atom)
        self._confidence = confidence

    @property
    def confidence(self) -> float:
        return self._confidence

    def __repr__(self) -> str:
        return "SoftFact({}, {}, conf={})".format(
            self.num, self.atom, self.confidence)


class SoftRule(Rule):
    """A soft (hypothesized) rule with confidence."""

    def __init__(self, num: int, head: str, body: str, confidence: float):
        super().__init__(num, head, body)
        self._confidence = confidence

    @property
    def confidence(self) -> float:
        return self._confidence

    def __repr__(self) -> str:
        return "SoftRule({}, {} :- {}, conf={})".format(
            self.num, self.head, self.body, self.confidence)


class DerivedFact(SoftFact):
    def __init__(self, num: int, atom: AtomicFormula, confidence: float, depth: int, rule: Rule):
        super().__init__(num, atom, confidence)
        self.depth = depth
        self.rule = rule
