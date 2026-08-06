"""Prolog command representations (facts and rules)."""

from typing import Optional, List, Tuple

from logic.logic import Term, AtomicFormula
from prolog.formula_parsing import (
    parse_prolog_to_formula,
    parse_body_to_formula, parse_predicate,
)


def canonical_atom_str(atom: str) -> str:
    """Spacing-independent form of a Prolog atom string.

    ``AtomicFormula.__str__`` writes no space after the argument separator
    (``p(a,b)``), whereas the KB ``.pl`` files in this repo write one
    (``p(a, b)``). Both parse to the same formula, so only this normalised form
    may be used to decide whether two atoms are THE SAME atom.
    """
    return str(parse_prolog_to_formula(atom))


class Fact:
    """A Prolog fact.

    Two string views, deliberately distinct:

    ``atom``            – the text as written (KB file / LLM output). This is what
                          gets rendered back by ``KnowledgeBase.to_text()`` and
                          into the LLM prompts, so it must preserve the author's
                          formatting.
    ``canonical_atom``  – ``str(self.formula)``; the identity key. Comparing raw
                          ``atom`` text instead let ``connected(a,b)`` and
                          ``connected(a, b)`` coexist as two separate soft facts,
                          which inflated ``soft_facts_proposed`` and pushed
                          ``recall_omitted`` above 1.0 in the 20260804 run.
    """

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
        # Computed once: every identity check goes through this, and re-parsing
        # per comparison would make add_soft_fact quadratic in parse calls.
        self._canonical_atom = str(self.formula)

    @property
    def atom(self) -> str:
        return self._atom_str

    @property
    def canonical_atom(self) -> str:
        """Identity key: spacing-independent rendering of this fact's atom."""
        return self._canonical_atom

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
