"""Prolog command representations (facts and rules)."""

from typing import Optional

from prolog_llm.prolog_utils import parse_predicate

from prolog.prolog_parsing import (
    parse_prolog_to_formula,
    parse_body_to_formula,
)

from logic.logic import (
    AtomicFormula,
    Formula,
    Conjunction,
)


class Fact:
    """A Prolog fact."""
    
    def __init__(self, num: int, atom: str):
        self.num = num
        self._atom_str = atom
        self.formula = parse_prolog_to_formula(atom)
        parsed = parse_predicate(atom)
        if parsed:
            self.functor, self.args = parsed
        else:
            self.functor, self.args = "", []
    
    @property
    def atom(self) -> str:
        return self._atom_str
    
    def __repr__(self) -> str:
        return "Fact({}, {})".format(self.num, self.atom)


class Rule:
    """A Prolog rule."""
    
    def __init__(self, num: int, head: str, body: str):
        self.num = num
        self._head_str = head
        self._body_str = body
        self.head_formula = parse_prolog_to_formula(head)
        self.body_formula = parse_body_to_formula(body)
        parsed = parse_predicate(head)
        if parsed:
            self.functor, self.args = parsed
        else:
            self.functor, self.args = "", []
    
    @property
    def head(self) -> str:
        return self._head_str
    
    @property
    def body(self) -> str:
        return self._body_str
    
    def __repr__(self) -> str:
        return "Rule({}, {} :- {})".format(self.num, self.head, self.body)
