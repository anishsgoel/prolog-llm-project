"""Knowledge base representation for Prolog."""

from typing import Dict, List, Set

from prolog.program_parsing import parse
from prolog.prolog_command import Fact, Rule, SoftFact, SoftRule


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
