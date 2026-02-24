"""Prolog BFS solvers as classes."""

import heapq
import os
import sys
from collections import deque
from itertools import count
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prolog_llm.knowledge_base import KnowledgeBase
from prolog_llm.soft_kb import SoftKB
from prolog_llm.prolog_utils import (
    check_exact_match,
    unify_with_fact,
    apply_bindings,
    find_matching_rules_only,
    get_subgoals,
    is_ground_atom,
)
import config


class PrologSolver:
    """
    Base class for Prolog BFS solvers.
    """
    
    def __init__(self, kb: KnowledgeBase, max_depth: Optional[int] = None):
        self.kb = kb
        self.max_depth = max_depth or config.DEFAULT_MAX_DEPTH
    
    def _create_queue_item(self, goal, remaining, path, depth):
        """Override in subclasses."""
        raise NotImplementedError


class HardKBCollector(PrologSolver):
    """
    Hard-KB BFS solver that collects metrics.
    Returns a single "blocking_atom" (ground) if possible.
    """
    
    def solve(self, goal: str) -> dict:
        """Run the solver and return result with metrics."""
        
        queue = deque([(goal, [], [], 0)])
        visited = set()
        
        expansions = 0
        best_blocking_atom = None
        best_blocking_depth = -1
        blocking_reason = None
        
        if config.VERBOSE:
            print(f"\n[COLLECT] Goal: {goal}")
            print("-" * 40)
        
        while queue:
            current, remaining, path, depth = queue.popleft()
            
            expansions += 1
            
            if depth >= self.max_depth:
                if is_ground_atom(current) and depth > best_blocking_depth:
                    best_blocking_atom = current
                    best_blocking_depth = depth
                    blocking_reason = "depth_cap"
                continue
            
            state = (current, tuple(remaining))
            if state in visited:
                continue
            visited.add(state)
            
            if config.VERBOSE:
                print(f"Depth {depth}: {current}")
                if remaining:
                    print(f"  Remaining: {remaining}")
            
            progress = False
            
            for fact in self.kb.facts:
                if check_exact_match(current, fact.atom):
                    if config.VERBOSE:
                        print(f"  ✓ Fact {fact.num} matches exactly!")
                    
                    new_path = path + [f"Fact {fact.num}"]
                    if not remaining:
                        if config.VERBOSE:
                            print(f"✓✓ SUCCESS at depth {depth + 1}")
                        return {
                            "success": True,
                            "proof_path": new_path,
                            "unresolved_atoms": set(),
                            "metrics": {
                                "hard_expansions": expansions,
                                "blocking_atom": None,
                                "blocking_depth": None,
                                "blocking_reason": None,
                            }
                        }
                    
                    queue.append((remaining[0], remaining[1:], new_path, depth + 1))
                    progress = True
                    break
            
            if progress:
                continue
            
            for fact in self.kb.facts:
                bindings = unify_with_fact(current, fact.atom)
                if bindings is None:
                    continue
                
                progress = True
                
                if config.VERBOSE:
                    print(f"  ✓ Fact {fact.num}: {fact.atom}")
                    print(f"    Bindings: {bindings}")
                
                instantiated = apply_bindings(remaining, bindings)
                new_path = path + [f"Fact {fact.num}"]
                
                if not instantiated:
                    if config.VERBOSE:
                        print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set(),
                        "metrics": {
                            "hard_expansions": expansions,
                            "blocking_atom": None,
                            "blocking_depth": None,
                            "blocking_reason": None,
                        }
                    }
                
                queue.append((instantiated[0], instantiated[1:], new_path, depth + 1))
            
            if progress:
                continue
            
            matching_rules = find_matching_rules_only(
                current, 
                [(r.num, r.head, r.body) for r in self.kb.rules]
            )
            if config.VERBOSE and matching_rules:
                print(f"  Matching rules: {matching_rules}")
            
            for rule_num in matching_rules:
                for rule in self.kb.rules:
                    if rule.num != rule_num:
                        continue
                    subgoals = get_subgoals(current, rule.head, rule.body)
                    if subgoals:
                        if config.VERBOSE:
                            print(f"  Rule {rule.num}: → {subgoals}")
                        progress = True
                        all_goals = subgoals + remaining
                        queue.append((all_goals[0], all_goals[1:], path + [f"Rule {rule.num}"], depth + 1))
                    break
            
            if not progress:
                if config.VERBOSE:
                    print(f"  ✗ No facts or rules apply to: {current}")
                
                if is_ground_atom(current) and depth > best_blocking_depth:
                    best_blocking_atom = current
                    best_blocking_depth = depth
                    blocking_reason = "no_progress"
        
        if config.VERBOSE:
            print("✗ FAILED (collect mode)")
        
        unresolved = set()
        if best_blocking_atom:
            unresolved.add(best_blocking_atom)
        
        return {
            "success": False,
            "proof_path": [],
            "unresolved_atoms": unresolved,
            "metrics": {
                "hard_expansions": expansions,
                "blocking_atom": best_blocking_atom,
                "blocking_depth": best_blocking_depth if best_blocking_atom else None,
                "blocking_reason": blocking_reason,
            }
        }


class SoftBFSSolver(PrologSolver):
    """
    Soft BFS with confidence-first priority + penalty.
    """
    
    def __init__(
        self, 
        kb: KnowledgeBase, 
        soft_kb: SoftKB,
        max_depth: Optional[int] = None,
        max_soft: Optional[int] = None,
        max_solutions: Optional[int] = None
    ):
        super().__init__(kb, max_depth)
        self.soft_kb = soft_kb
        self.max_soft = max_soft
        self.max_solutions = max_solutions or config.SOFT_BFS_MAX_SOLUTIONS
    
    def solve(self, goal: str) -> dict:
        """Run the soft BFS solver."""
        
        pq = []
        tie = count()
        
        def push_state(current, remaining, path, depth, soft_cost, min_conf, penalty_sum):
            if depth > self.max_depth:
                return
            heapq.heappush(
                pq,
                (soft_cost, -min_conf, penalty_sum, depth, next(tie),
                 current, remaining, path, min_conf, penalty_sum)
            )
        
        push_state(goal, [], [], 0, 0, 1.0, 0.0)
        
        best_seen = {}
        
        def dominated(current, remaining, soft_cost, min_conf, penalty_sum, depth):
            key = (current, tuple(remaining))
            new = (soft_cost, -min_conf, penalty_sum, depth)
            old = best_seen.get(key)
            if old is None:
                best_seen[key] = new
                return False
            
            if (old[0], old[1], old[2]) < (new[0], new[1], new[2]):
                return True
            if (old[0], old[1], old[2]) == (new[0], new[1], new[2]) and old[3] <= new[3]:
                return True
            
            best_seen[key] = new
            return False
        
        def make_success_result(final_path, final_soft_cost, final_min_conf, final_depth, final_penalty_sum):
            used_soft = []
            for step in final_path:
                if step.startswith("SoftFact"):
                    parts = step.split()
                    if len(parts) >= 2:
                        try:
                            num = int(parts[1])
                        except ValueError:
                            num = None
                        used_soft.append(("fact", num))
                elif step.startswith("SoftRule"):
                    parts = step.split()
                    if len(parts) >= 2:
                        try:
                            num = int(parts[1])
                        except ValueError:
                            num = None
                        used_soft.append(("rule", num))
            
            return {
                "success": True,
                "proof_path": final_path,
                "used_soft_clauses": used_soft,
                "soft_cost": final_soft_cost,
                "min_conf": final_min_conf if final_soft_cost > 0 else None,
                "penalty_sum": final_penalty_sum,
                "depth": final_depth,
            }
        
        solutions = []
        
        def maybe_add_solution(res, soft_cost, min_conf, penalty_sum):
            solutions.append((soft_cost, -min_conf, penalty_sum, res))
            solutions.sort(key=lambda x: (x[0], x[1], x[2]))
            if len(solutions) > self.max_solutions:
                solutions.pop()
        
        while pq:
            soft_cost, neg_min_conf, penalty_sum, depth, _, current, remaining, path, min_conf, penalty_sum = heapq.heappop(pq)
            
            if config.VERBOSE:
                print(f"Depth {depth}: {current}")
                print(f"  Priority key: (soft_cost={soft_cost}, -min_conf={neg_min_conf:.3f}, penalty={penalty_sum:.3f}, depth={depth})")
                if remaining:
                    print(f"  Remaining: {remaining}")
            
            if dominated(current, remaining, soft_cost, min_conf, penalty_sum, depth):
                continue
            
            for fact in self.kb.facts:
                if check_exact_match(current, fact.atom):
                    if config.VERBOSE:
                        print(f"  ✓ Hard Fact {fact.num} matches exactly: {fact.atom}")
                    new_path = path + [f"HardFact {fact.num}"]
                    
                    if not remaining:
                        res = make_success_result(new_path, soft_cost, min_conf, depth + 1, penalty_sum)
                        maybe_add_solution(res, soft_cost, min_conf, penalty_sum)
                        break
                    
                    push_state(remaining[0], remaining[1:], new_path, depth + 1, soft_cost, min_conf, penalty_sum)
                    break
            
            for fact in self.kb.facts:
                bindings = unify_with_fact(current, fact.atom)
                if bindings is None:
                    continue
                
                if config.VERBOSE:
                    print(f"  ✓ Hard Fact {fact.num} unifies: {fact.atom}")
                    print(f"    Bindings: {bindings}")
                
                instantiated = apply_bindings(remaining, bindings)
                new_path = path + [f"HardFact {fact.num}"]
                
                if not instantiated:
                    res = make_success_result(new_path, soft_cost, min_conf, depth + 1, penalty_sum)
                    maybe_add_solution(res, soft_cost, min_conf, penalty_sum)
                    continue
                
                push_state(instantiated[0], instantiated[1:], new_path, depth + 1, soft_cost, min_conf, penalty_sum)
            
            matching_hard_rules = find_matching_rules_only(
                current,
                [(r.num, r.head, r.body) for r in self.kb.rules]
            )
            if config.VERBOSE and matching_hard_rules:
                print(f"  Matching hard rules: {matching_hard_rules}")
            
            for rule_num in matching_hard_rules:
                for rule in self.kb.rules:
                    if rule.num != rule_num:
                        continue
                    
                    subgoals = get_subgoals(current, rule.head, rule.body)
                    if not subgoals:
                        continue
                    
                    if config.VERBOSE:
                        print(f"  Hard Rule {rule.num}: {rule.head} :- {rule.body}")
                        print(f"    → {subgoals}")
                    
                    all_goals = subgoals + remaining
                    push_state(all_goals[0], all_goals[1:], path + [f"HardRule {rule.num}"], depth + 1,
                               soft_cost, min_conf, penalty_sum)
                    break
            
            for sf in self.soft_kb.facts:
                if self.max_soft is not None and soft_cost >= self.max_soft:
                    break
                
                bindings = unify_with_fact(current, sf.atom)
                if bindings is None:
                    continue
                
                new_soft_cost = soft_cost + 1
                new_min_conf = min(min_conf, sf.confidence)
                new_penalty_sum = penalty_sum + sf.penalty
                
                if config.VERBOSE:
                    print(f"  ✓ Soft Fact {sf.num} unifies: {sf.atom}")
                    print(f"    Bindings: {bindings}, conf={sf.confidence:.3f}, penalty={sf.penalty:.3f}")
                    print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}, new penalty: {new_penalty_sum:.3f}")
                
                instantiated = apply_bindings(remaining, bindings)
                new_path = path + [f"SoftFact {sf.num} (conf={sf.confidence:.3f},pen={sf.penalty:.3f})"]
                
                if not instantiated:
                    res = make_success_result(new_path, new_soft_cost, new_min_conf, depth + 1, new_penalty_sum)
                    maybe_add_solution(res, new_soft_cost, new_min_conf, new_penalty_sum)
                    continue
                
                push_state(instantiated[0], instantiated[1:], new_path, depth + 1,
                           new_soft_cost, new_min_conf, new_penalty_sum)
            
            matching_soft_rules = find_matching_rules_only(
                current,
                self.soft_kb.get_rules_for_matching()
            )
            if config.VERBOSE and matching_soft_rules:
                print(f"  Matching soft rules: {matching_soft_rules}")
            
            for rule_num in matching_soft_rules:
                if self.max_soft is not None and soft_cost >= self.max_soft:
                    break
                
                for sr in self.soft_kb.rules:
                    if sr.num != rule_num:
                        continue
                    
                    subgoals = get_subgoals(current, sr.head, sr.body)
                    if not subgoals:
                        continue
                    
                    new_soft_cost = soft_cost + 1
                    new_min_conf = min(min_conf, sr.confidence)
                    new_penalty_sum = penalty_sum + sr.penalty
                    
                    if config.VERBOSE:
                        print(f"  Soft Rule {sr.num}: {sr.head} :- {sr.body}")
                        print(f"    → {subgoals}, conf={sr.confidence:.3f}, penalty={sr.penalty:.3f}")
                        print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}, new penalty: {new_penalty_sum:.3f}")
                    
                    all_goals = subgoals + remaining
                    new_path = path + [f"SoftRule {sr.num} (conf={sr.confidence:.3f},pen={sr.penalty:.3f})"]
                    
                    push_state(all_goals[0], all_goals[1:], new_path, depth + 1,
                               new_soft_cost, new_min_conf, new_penalty_sum)
                    break
        
        if solutions:
            return solutions[0][3]
        
        if config.VERBOSE:
            print("✗ PRIORITY SOFT-BFS FAILED (no proof found even with soft KB)")
        return {
            "success": False,
            "proof_path": [],
            "used_soft_clauses": [],
            "soft_cost": None,
            "min_conf": None,
            "penalty_sum": None,
            "depth": None,
        }
