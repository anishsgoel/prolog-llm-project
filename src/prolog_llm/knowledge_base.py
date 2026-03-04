"""Knowledge base representation for Prolog."""

import os
import re
import sys
from typing import Dict, List, Set, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prolog_llm.prolog_utils import (
    parse_predicate,
    split_inline_comment,
    is_variable,
)


class Fact:
    """A Prolog fact."""
    
    def __init__(self, num: int, atom: str):
        self.num = num
        self.atom = atom
        parsed = parse_predicate(atom)
        if parsed:
            self.functor, self.args = parsed
        else:
            self.functor, self.args = "", []
    
    def __repr__(self) -> str:
        return f"Fact({self.num}, {self.atom})"


class Rule:
    """A Prolog rule."""
    
    def __init__(self, num: int, head: str, body: str):
        self.num = num
        self.head = head
        self.body = body
        parsed = parse_predicate(head)
        if parsed:
            self.functor, self.args = parsed
        else:
            self.functor, self.args = "", []
    
    def __repr__(self) -> str:
        return f"Rule({self.num}, {self.head} :- {self.body})"


class KnowledgeBase:
    """
    Represents a Prolog knowledge base with facts and rules.
    Provides parsing, querying, and manipulation methods.
    """
    
    def __init__(self, kb_text: str = ""):
        self.facts: List[Fact] = []
        self.rules: List[Rule] = []
        self.predicate_comments: Dict[str, str] = {}
        self._adjacency: Dict[str, List[str]] = {}
        self._stations: Set[str] = set()
        
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
                
                key = "{}/{}".format(rule.functor, len(rule.args))
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
                
                key = "{}/{}".format(fact.functor, len(fact.args))
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
    def adjacency(self) -> Dict[str, List[str]]:
        """Return adjacency dict for connected/2 facts."""
        return self._adjacency
    
    @property
    def stations(self) -> Set[str]:
        """Return set of all stations."""
        return self._stations
    
    @property
    def edges(self) -> Set[tuple]:
        """Return set of (src, dst) tuples for connected/2 facts."""
        edges = set()
        for fact in self.facts:
            if fact.functor == "connected" and len(fact.args) == 2:
                a, b = fact.args[0].strip(), fact.args[1].strip()
                edges.add((a, b))
        return edges
    
    def get_facts_by_functor(self, functor: str, arity: int) -> List[Fact]:
        """Get all facts matching functor/arity."""
        return [f for f in self.facts if f.functor == functor and len(f.args) == arity]
    
    def get_rules_by_functor(self, functor: str, arity: int) -> List[Rule]:
        """Get all rules matching functor/arity."""
        return [r for r in self.rules if r.functor == functor and len(r.args) == arity]
    
    def omit_facts(self, omit_numbers: Set[int]) -> "KnowledgeBase":
        """Create a new KB with specified fact numbers removed."""
        new_kb = KnowledgeBase()
        new_kb.facts = [f for f in self.facts if f.num not in omit_numbers]
        new_kb.rules = self.rules[:]
        new_kb.predicate_comments = self.predicate_comments.copy()
        new_kb._build_adjacency()
        return new_kb
    
    def omit_facts_from_text(self, kb_text: str, omit_numbers: Set[int]) -> "KnowledgeBase":
        """Parse KB text, remove specified fact numbers, return new KB."""
        kb = KnowledgeBase(kb_text)
        return kb.omit_facts(omit_numbers)
    
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
