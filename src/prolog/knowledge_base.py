"""Knowledge base representation for Prolog."""

from typing import Dict, List, Optional, Set, Tuple

from prolog.program_parsing import parse
from prolog.prolog_command import Fact, Rule, SoftFact, SoftRule, canonical_atom_str


class KnowledgeBase:
    """
    Represents a Prolog knowledge base with facts and rules.
    Provides parsing, querying, and manipulation methods.
    """
    
    def __init__(self, kb_text: str = ""):
        self.facts: List[Fact] = []
        self.rules: List[Rule] = []
        self.predicate_comments: Dict[str, str] = {}

        if kb_text:
            self.parse(kb_text)
    
    def parse(self, kb_text: str) -> None:
        """Parse KB text into facts and rules."""
        self.facts, self.rules, self.predicate_comments = parse(kb_text)

    def get_facts_by_functor(self, functor: str, arity: int) -> List[Fact]:
        """Get all facts matching functor/arity."""
        return [f for f in self.facts if f.formula.pred.name == functor and len(f.formula.pred.arity) == arity]
    
    def get_rules_by_functor(self, functor: str, arity: int) -> List[Rule]:
        """Get all rules matching functor/arity."""
        return [r for r in self.rules if r.head_formula.pred.name == functor and len(r.head_formula.pred.arity) == arity]
    
    def omit_facts(self, omit_numbers: Set[int]) -> "KnowledgeBase":
        """Create a new KB with specified fact numbers removed."""
        new_kb = KnowledgeBase()
        new_kb.facts = [f for f in self.facts if f.num not in omit_numbers]
        new_kb.rules = self.rules[:]
        new_kb.predicate_comments = self.predicate_comments.copy()
        return new_kb
    
    def to_text(self) -> str:
        """Convert KB back to text format."""
        lines = []
        all_clauses = []
        
        for fact in self.facts:
            all_clauses.append((fact.num, "{}.".format(fact.atom)))
        for rule in self.rules:
            all_clauses.append((rule.num, "{} :- {}.".format(rule.head, rule.body)))
        
        all_clauses.sort(key=lambda x: x[0])
        for num, clause in all_clauses:
            lines.append("{}. {}".format(num, clause))
        
        return "\n".join(lines)
    
    def get_max_line_number(self) -> int:
        """Get the highest line number in the KB."""
        max_num = 0
        for fact in self.facts:
            max_num = max(max_num, fact.num)
        for rule in self.rules:
            max_num = max(max_num, rule.num)
        return max_num
    
    def __repr__(self) -> str:
        return "KnowledgeBase(facts={}, rules={})".format(len(self.facts), len(self.rules))


class SoftKnowledgeBase(KnowledgeBase):
    """Knowledge base containing soft facts and soft rules."""

    def __init__(self, hard: KnowledgeBase):
        super().__init__(None)
        self.facts: List[SoftFact] = [SoftFact(f.num, f.atom, 1.0) for f in hard.facts]
        self.rules: List[SoftRule] = [SoftRule(r.num, r.head, r.body, 1.0) for r in hard.rules]
        self.predicate_comments: Dict[str, str] = hard.predicate_comments

    def copy(self) -> "SoftKnowledgeBase":
        """Create a shallow copy of the soft knowledge base."""
        new_kb = SoftKnowledgeBase.__new__(SoftKnowledgeBase)
        new_kb.facts = [SoftFact(f.num, f.atom, f.confidence) for f in self.facts]
        new_kb.rules = [SoftRule(r.num, r.head, r.body, r.confidence) for r in self.rules]
        new_kb.predicate_comments = self.predicate_comments.copy()
        return new_kb

    def upsert_soft_fact(self, num: int, atom: str,
                         confidence: float) -> Tuple[Optional[SoftFact], bool]:
        """Insert a soft fact, or raise the confidence of the matching one.

        Returns ``(fact, appended)``:

        ``(None, False)``   nothing changed — already present at >= confidence
        ``(fact, False)``   an existing fact's confidence was raised to this one
        ``(fact, True)``    a new fact was appended

        Returning the fact, rather than only a bool, is what lets a caller know
        *which* fact it affected. ``apply_clauses`` used to reconstruct that by
        snapshotting ``len(self.facts)``, rescanning for a fact with a matching
        atom and confidence, and falling back to ``self.facts[-1]`` — a guess for
        a question this method can simply answer.

        Matching is by CANONICAL atom, not by raw text: an LLM that proposes
        ``connected(a,b)`` after ``connected(a, b)`` must update the fact it
        already put there rather than append a second copy. The raw text is
        still what gets stored and rendered (see :class:`Fact`), so prompts and
        ``to_text()`` are unaffected.
        """
        try:
            key = canonical_atom_str(atom)
        except Exception:
            # Unparseable atom: keep the old raw behaviour. SoftFact() below
            # parses the same string and will surface the real error.
            key = atom

        for fact in self.facts:
            if fact.canonical_atom != key:
                continue
            if confidence > fact.confidence:
                fact._confidence = confidence
                return fact, False
            return None, False

        new_fact = SoftFact(num, atom, confidence)
        self.facts.append(new_fact)
        return new_fact, True

    def add_soft_fact(self, num: int, atom: str, confidence: float) -> bool:
        """Did-anything-change wrapper around :meth:`upsert_soft_fact`.

        Kept because it is the historical public entry point; new code should
        prefer ``upsert_soft_fact``, which also says which fact changed.
        """
        fact, _ = self.upsert_soft_fact(num, atom, confidence)
        return fact is not None
