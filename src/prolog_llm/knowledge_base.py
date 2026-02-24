"""Knowledge base representation for Prolog."""

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prolog_llm.prolog_utils import (
    parse_predicate,
    split_inline_comment,
    is_variable,
)


@dataclass
class Fact:
    """A Prolog fact."""
    num: int
    atom: str
    functor: str = field(init=False)
    args: list = field(init=False)
    
    def __post_init__(self):
        parsed = parse_predicate(self.atom)
        if parsed:
            self.functor, self.args = parsed
        else:
            self.functor, self.args = "", []


@dataclass
class Rule:
    """A Prolog rule."""
    num: int
    head: str
    body: str
    functor: str = field(init=False)
    args: list = field(init=False)
    
    def __post_init__(self):
        parsed = parse_predicate(self.head)
        if parsed:
            self.functor, self.args = parsed
        else:
            self.functor, self.args = "", []


@dataclass 
class PredicateComment:
    """Documentation for a predicate."""
    predicate: str  # e.g., "connected/2"
    comment: str


class KnowledgeBase:
    """
    Represents a Prolog knowledge base with facts and rules.
    Provides parsing, querying, and manipulation methods.
    """
    
    def __init__(self, kb_text: str = ""):
        self.facts: list[Fact] = []
        self.rules: list[Rule] = []
        self.predicate_comments: dict[str, str] = {}
        self._adjacency: dict[str, list[str]] = {}
        self._stations: set[str] = set()
        
        if kb_text:
            self.parse(kb_text)
    
    def parse(self, kb_text: str) -> None:
        """Parse KB text into facts and rules."""
        self.facts = []
        self.rules = []
        self.predicate_comments = {}
        
        pending_comments = []
        
        for line in kb_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("#"):
                pending_comments.append(line.lstrip("#").strip())
                continue
            
            m = re.match(r"^(\d+)\.\s*(.+)$", line)
            if not m:
                continue
            
            num = int(m.group(1))
            content_raw = m.group(2).strip()
            content, inline_comment = split_inline_comment(content_raw)
            
            if not content:
                continue
            
            if ":-" in content:
                head, body = content.split(":-", 1)
                rule = Rule(
                    num=num,
                    head=head.strip(),
                    body=body.strip().rstrip(".")
                )
                self.rules.append(rule)
                
                key = f"{rule.functor}/{len(rule.args)}"
                if pending_comments or inline_comment:
                    combined = []
                    if pending_comments:
                        combined.append(" ".join(pending_comments))
                    if inline_comment:
                        combined.append(inline_comment)
                    self.predicate_comments[key] = " ".join(combined).strip()
            else:
                fact = Fact(num=num, atom=content.rstrip("."))
                self.facts.append(fact)
                
                key = f"{fact.functor}/{len(fact.args)}"
                if pending_comments or inline_comment:
                    combined = []
                    if pending_comments:
                        combined.append(" ".join(pending_comments))
                    if inline_comment:
                        combined.append(inline_comment)
                    self.predicate_comments[key] = " ".join(combined).strip()
            
            pending_comments = []
        
        self._build_adjacency()
    
    def _build_adjacency(self) -> None:
        """Build adjacency list for connected/2 facts."""
        self._adjacency = {}
        self._stations = set()
        
        for fact in self.facts:
            if fact.functor == "connected" and len(fact.args) == 2:
                a, b = fact.args[0].strip(), fact.args[1].strip()
                self._adjacency.setdefault(a, []).append(b)
                self._stations.add(a)
                self._stations.add(b)
    
    @property
    def adjacency(self) -> dict[str, list[str]]:
        """Return adjacency dict for connected/2 facts."""
        return self._adjacency
    
    @property
    def stations(self) -> set[str]:
        """Return set of all stations."""
        return self._stations
    
    @property
    def edges(self) -> set[tuple[str, str]]:
        """Return set of (src, dst) tuples for connected/2 facts."""
        edges = set()
        for fact in self.facts:
            if fact.functor == "connected" and len(fact.args) == 2:
                a, b = fact.args[0].strip(), fact.args[1].strip()
                edges.add((a, b))
        return edges
    
    def get_facts_by_functor(self, functor: str, arity: int) -> list[Fact]:
        """Get all facts matching functor/arity."""
        return [f for f in self.facts if f.functor == functor and len(f.args) == arity]
    
    def get_rules_by_functor(self, functor: str, arity: int) -> list[Rule]:
        """Get all rules matching functor/arity."""
        return [r for r in self.rules if r.functor == functor and len(r.args) == arity]
    
    def omit_facts(self, omit_numbers: set[int]) -> "KnowledgeBase":
        """Create a new KB with specified fact numbers removed."""
        new_kb = KnowledgeBase()
        new_kb.facts = [f for f in self.facts if f.num not in omit_numbers]
        new_kb.rules = self.rules.copy()
        new_kb.predicate_comments = self.predicate_comments.copy()
        new_kb._build_adjacency()
        return new_kb
    
    def to_text(self) -> str:
        """Convert KB back to text format."""
        lines = []
        all_clauses = []
        
        for fact in self.facts:
            all_clauses.append((fact.num, f"{fact.atom}."))
        for rule in self.rules:
            all_clauses.append((rule.num, f"{rule.head} :- {rule.body}."))
        
        all_clauses.sort(key=lambda x: x[0])
        for num, clause in all_clauses:
            lines.append(f"{num}. {clause}")
        
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
        return f"KnowledgeBase(facts={len(self.facts)}, rules={len(self.rules)})"
