"""Hypothesis generator using LLM."""

import json
import os
import re
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prolog_llm.prolog_utils import (
    extract_first_json,
)
from prolog.prolog_utils import is_variable
from prolog.formula_parsing import parse_predicate
from prolog_llm.llm_interface import LLMInterface, llm_json_only
from prolog.knowledge_base import KnowledgeBase
from prolog_llm.soft_kb import SoftKB
import config


class HypothesisGenerator:
    """
    Generates background hypotheses using LLM for failed Prolog goals.
    Provides caching and configurable parameters.
    """
    
    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        max_atoms: Optional[int] = None,
        max_hyp_per_atom: Optional[int] = None,
        prompt_fact_limit: Optional[int] = None,
    ):
        """
        Initialize the hypothesis generator.
        
        Args:
            llm: LLM interface to use (creates default if None)
            max_atoms: Maximum atoms to query about
            max_hyp_per_atom: Maximum hypotheses per atom
            prompt_fact_limit: Maximum facts to include in prompt
        """
        self.llm = llm or LLMInterface()
        self.max_atoms = max_atoms if max_atoms is not None else config.HYP_MAX_ATOMS
        self.max_hyp_per_atom = max_hyp_per_atom if max_hyp_per_atom is not None else config.HYP_MAX_HYP_PER_ATOM
        self.prompt_fact_limit = prompt_fact_limit if prompt_fact_limit is not None else config.HYP_PROMPT_FACT_LIMIT
        self._cache: Dict[Tuple, List[Dict[str, Any]]] = {}
    
    def _kb_signature(self, kb: KnowledgeBase) -> str:
        """Generate signature for KB caching."""
        parts = []
        for fact in kb.facts:
            if fact.functor in ("connected", "reachable"):
                parts.append(fact.atom)
        return str(hash("\n".join(parts)))
    
    def _extract_constants(self, term: str) -> Set[str]:
        """Extract constants from a term."""
        p = parse_predicate(term.strip().rstrip("."))
        if not p:
            return set()
        _, args = p
        return {a.strip() for a in args if a.strip() and not is_variable(a.strip())}
    
    def _shortest_path(self, adj: Dict[str, List[str]], src: str, dst: str, max_depth: int) -> Optional[int]:
        """Find shortest path length bounded by max_depth."""
        if src == dst:
            return 0
        q = deque([(src, 0)])
        seen = {src}
        while q:
            node, d = q.popleft()
            if d >= max_depth:
                continue
            for nb in adj.get(node, []):
                if nb == dst:
                    return d + 1
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        return None
    
    def _infer_preferred_atom(self, goal: str, adj: Dict[str, List[str]]) -> Optional[str]:
        """Infer preferred atom for chain goals."""
        gp = parse_predicate(goal)
        if not gp or len(gp[1]) != 2:
            return None
        
        goal_src, goal_dst = gp[1][0].strip(), gp[1][1].strip()
        
        last = goal_src
        seen = set()
        while last in adj and len(adj[last]) == 1 and last not in seen:
            seen.add(last)
            last = adj[last][0]
        
        if last and goal_dst:
            return f"connected({last}, {goal_dst})"
        return None
    
    def _select_atoms(self, unresolved_atoms: Set[str], preferred_atom: Optional[str]) -> List[str]:
        """Select which atoms to query about."""
        atom_list = sorted(unresolved_atoms)
        
        ground = []
        for atom in atom_list:
            atom = atom.strip()
            if not atom or "(" not in atom or ")" not in atom:
                continue
            inside = atom.split("(", 1)[1].rsplit(")", 1)[0]
            if re.search(r"\b[A-Z_]\w*\b", inside):
                continue
            ground.append(atom)
        
        if preferred_atom:
            ground = [preferred_atom] + [a for a in ground if a != preferred_atom]
        
        return ground[:self.max_atoms]
    
    def _normalize_clause(self, clause: str) -> Optional[str]:
        """Normalize and validate a clause."""
        if not clause:
            return None
        clause = clause.strip()
        if not clause.endswith("."):
            clause += "."
        atom_str = clause.rstrip(".").strip()
        p = parse_predicate(atom_str)
        if not (p and p[0] == "connected" and len(p[1]) == 2):
            return None
        u, v = p[1][0].strip(), p[1][1].strip()
        if u == v:
            return None
        return f"connected({u}, {v})."
    
    def generate(
        self,
        goal: str,
        kb: KnowledgeBase,
        hard_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate background hypotheses for a failed goal.
        
        Args:
            goal: The Prolog goal that failed
            kb: The knowledge base
            hard_result: Result from hard KB solve
            
        Returns:
            List of hypothesis dicts with clause, confidence, etc.
        """
        unresolved_atoms = hard_result.get("unresolved_atoms", set()) or set()
        if not unresolved_atoms:
            return []
        
        adj = kb.adjacency
        edges = kb.edges
        stations = kb.stations
        
        stations_aug = set(stations)
        stations_aug |= self._extract_constants(goal)
        for a in unresolved_atoms:
            stations_aug |= self._extract_constants(a)
        
        preferred_atom = self._infer_preferred_atom(goal, adj)
        atoms = self._select_atoms(unresolved_atoms, preferred_atom)
        
        if not atoms:
            print("[HypothesisGenerator] No suitable ground atoms.")
            return []
        
        cache_key = (
            self._kb_signature(kb),
            goal,
            tuple(atoms),
            self.max_hyp_per_atom,
            self.prompt_fact_limit,
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        facts_for_prompt = [
            f"connected({a}, {b})." 
            for a, b in edges
        ][:self.prompt_fact_limit]
        
        needed = {"connected/2"}
        for a in atoms:
            p = parse_predicate(a)
            if p:
                needed.add(f"{p[0]}/{len(p[1])}")
        
        semantic_lines = []
        for k in sorted(needed):
            if k in kb.predicate_comments:
                semantic_lines.append(f"- {k}: {kb.predicate_comments[k]}")
        semantic_hint_block = "\n".join(semantic_lines) if semantic_lines else "(none)"
        
        atoms_block = "\n".join([f"- {a}" for a in atoms])
        
        prompt = f"""
You are a cautious Prolog expert. You propose missing FACTS only.

GOAL:
  {goal}

HARD KB SNAPSHOT (partial; do NOT assume any other edges exist):
Connected facts (sample):
{chr(10).join(facts_for_prompt)}

Predicate semantics (ground truth):
{semantic_hint_block}

The proof failed because these subgoals could not be proven:
{atoms_block}

Task:
For each failed subgoal above, propose up to {self.max_hyp_per_atom} additional Prolog FACTS
that are likely true and would help prove the GOAL.

Constraints:
- Output ONLY connected/2 FACTS. No rules.
- Each fact MUST end with a period.
- Prefer using only station atoms that appear in the snapshot or the GOAL.
- If a suggested edge might be a "shortcut" (skips intermediate stops), lower confidence.

Return ONLY valid JSON in exactly this shape:

{{
  "by_atom": {{
    "connected(42nd_street, bryant_park)": [
      {{"clause":"connected(grand_central, bryant_park).","confidence":0.95}},
      {{"clause":"connected(42nd_street, bryant_park).","confidence":0.55}}
    ]
  }}
}}
""".strip()
        
        repair_schema = f"""{{
  "by_atom": {{
    "{atoms[0]}": [
      {{"clause":"connected(grand_central, bryant_park).","confidence":0.95}},
      {{"clause":"connected(42nd_street, bryant_park).","confidence":0.55}}
    ]
  }}
}}"""
        
        raw = llm_json_only(prompt, repair_schema).strip()
        if config.DEBUG:
            print("\n[DEBUG HypothesisGenerator] raw:\n", raw)
        
        try:
            data = json.loads(extract_first_json(raw))
        except Exception as e:
            print("[HypothesisGenerator] JSON parse error:", e)
            print("Raw LLM output:", raw[:800])
            self._cache[cache_key] = []
            return []
        
        by_atom = data.get("by_atom", {})
        if not isinstance(by_atom, dict):
            self._cache[cache_key] = []
            return []
        
        accepted = []
        for atom_key, proposals in by_atom.items():
            if not isinstance(proposals, list):
                continue
            
            for item in proposals[:self.max_hyp_per_atom]:
                if not isinstance(item, dict):
                    continue
                
                clause = self._normalize_clause(item.get("clause", ""))
                if clause is None:
                    continue
                
                try:
                    conf = float(item.get("confidence", 0.0))
                except Exception:
                    conf = 0.0
                conf = max(0.0, min(1.0, conf))
                
                p = parse_predicate(clause.rstrip("."))
                u, v = p[1][0].strip(), p[1][1].strip()
                
                if (u, v) in edges:
                    continue
                
                d = self._shortest_path(adj, u, v, config.DEFAULT_MAX_DEPTH_SHORTCUT)
                is_shortcut = d is not None and d >= 2
                if is_shortcut:
                    continue
                
                unknown_station = bool(stations_aug) and (u not in stations_aug or v not in stations_aug)
                
                accepted.append({
                    "clause": clause,
                    "confidence": conf,
                    "from_atom": atom_key,
                    "is_shortcut": is_shortcut,
                    "shortcut_len": d,
                    "unknown_station": unknown_station,
                })
        
        dedup = {}
        for h in accepted:
            key = h["clause"]
            if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
                dedup[key] = h
        
        result = list(dedup.values())
        self._cache[cache_key] = result
        return result
    
    def generate_soft_kb(
        self,
        goal: str,
        kb: KnowledgeBase,
        hard_result: Dict[str, Any],
    ) -> SoftKB:
        """
        Generate hypotheses and create a SoftKB.
        
        Args:
            goal: The Prolog goal that failed
            kb: The knowledge base
            hard_result: Result from hard KB solve
            
        Returns:
            SoftKB with generated hypotheses
        """
        hypotheses = self.generate(goal, kb, hard_result)
        return SoftKB.from_hypotheses(hypotheses, kb.to_text())
    
    def clear_cache(self) -> None:
        """Clear the hypothesis cache."""
        self._cache.clear()
    
    def __repr__(self) -> str:
        return f"HypothesisGenerator(max_atoms={self.max_atoms}, max_hyp_per_atom={self.max_hyp_per_atom})"


# Backwards compatibility function
def generate_background_hypotheses_fast(
    goal: str,
    kb: str,
    hard_result: dict,
    predicate_comments: dict,
    max_atoms: Optional[int] = None,
    max_hyp_per_atom: Optional[int] = None,
    prompt_fact_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Backwards-compatible function wrapper."""
    kb_obj = KnowledgeBase(kb)
    generator = HypothesisGenerator(
        max_atoms=max_atoms,
        max_hyp_per_atom=max_hyp_per_atom,
        prompt_fact_limit=prompt_fact_limit,
    )
    return generator.generate(goal, kb_obj, hard_result)


def attach_hypotheses_to_kb(kb: str, hypotheses: List[Dict[str, Any]]) -> SoftKB:
    """Backwards-compatible function wrapper."""
    return SoftKB.from_hypotheses(hypotheses, kb)
