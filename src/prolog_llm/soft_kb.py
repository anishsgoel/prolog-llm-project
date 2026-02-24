"""Soft KB for handling hypothesized facts and rules with confidence."""

import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prolog_llm.knowledge_base import KnowledgeBase
from prolog_llm.prolog_utils import parse_predicate
import config


@dataclass
class SoftFact:
    """A soft (hypothesized) fact with confidence and penalty."""
    num: int
    atom: str
    confidence: float
    penalty: float
    
    @property
    def functor(self) -> str:
        parsed = parse_predicate(self.atom)
        return parsed[0] if parsed else ""
    
    @property
    def args(self) -> list:
        parsed = parse_predicate(self.atom)
        return parsed[1] if parsed else []


@dataclass
class SoftRule:
    """A soft (hypothesized) rule with confidence and penalty."""
    num: int
    head: str
    body: str
    confidence: float
    penalty: float
    
    @property
    def functor(self) -> str:
        parsed = parse_predicate(self.head)
        return parsed[0] if parsed else ""
    
    @property
    def args(self) -> list:
        parsed = parse_predicate(self.head)
        return parsed[1] if parsed else []


class SoftKB:
    """
    Represents a soft knowledge base with hypothesized facts and rules.
    Used for storing LLM-generated hypotheses with confidence scores.
    """
    
    def __init__(self):
        self.facts: list[SoftFact] = []
        self.rules: list[SoftRule] = []
    
    @classmethod
    def from_hypotheses(cls, hypotheses: list[dict], hard_kb_text: str) -> "SoftKB":
        """
        Create SoftKB from list of hypothesis dicts.
        
        Args:
            hypotheses: List of dicts with 'clause', 'confidence', 'unknown_station' keys
            hard_kb_text: Original KB text (to find max line number)
        """
        kb = KnowledgeBase(hard_kb_text)
        max_num = kb.get_max_line_number()
        next_num = max_num + 1
        
        soft_kb = cls()
        
        for h in hypotheses:
            clause = (h.get("clause") or "").strip()
            if not clause:
                continue
            
            conf = float(h.get("confidence", 0.0))
            if not clause.endswith("."):
                clause += "."
            
            penalty = 0.0
            if h.get("unknown_station"):
                penalty += config.SOFT_PENALTY_UNKNOWN_STATION
            
            if ":-" not in clause:
                atom = clause.rstrip(".").strip()
                soft_kb.facts.append(SoftFact(
                    num=next_num,
                    atom=atom,
                    confidence=conf,
                    penalty=penalty
                ))
            else:
                head, body_str = clause.split(":-", 1)
                soft_kb.rules.append(SoftRule(
                    num=next_num,
                    head=head.strip(),
                    body=body_str.strip(),
                    confidence=conf,
                    penalty=penalty
                ))
            
            next_num += 1
        
        return soft_kb
    
    def get_facts_for_matching(self) -> list[tuple[int, str]]:
        """Get facts as (num, atom) tuples for rule matching."""
        return [(f.num, f.atom) for f in self.facts]
    
    def get_rules_for_matching(self) -> list[tuple[int, str, str]]:
        """Get rules as (num, head, body) tuples for rule matching."""
        return [(r.num, r.head, r.body) for r in self.rules]
    
    def __repr__(self) -> str:
        return f"SoftKB(facts={len(self.facts)}, rules={len(self.rules)})"
    
    def __len__(self) -> int:
        return len(self.facts) + len(self.rules)
