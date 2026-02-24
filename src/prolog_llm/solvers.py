"""BFS Prolog solvers (hard and soft)."""

import os
import re
import sys
from collections import deque
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prolog_llm.prolog_utils import (
    check_exact_match,
    parse_predicate,
    strip_inline_comment,
    split_inline_comment,
    unify_with_fact,
    apply_bindings,
    find_matching_rules_only,
    get_subgoals,
    is_ground_atom,
)
import config


def parse_kb(kb: str):
    """Parse KB into facts and rules."""
    facts = []
    rules = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        line = strip_inline_comment(line).strip()
        if not line:
            continue

        match = re.match(r"^(\d+)\.\s*(.+)$", line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ":-" in content:
                head, body = content.split(":-", 1)
                rules.append((num, head.strip(), body.strip().rstrip(".")))
            else:
                facts.append((num, content.rstrip(".")))

    return facts, rules


def bfs_prolog_collect(goal: str, kb: str, max_depth: Optional[int] = None):
    """
    Hard-KB BFS solver that collects metrics.
    Returns a single "blocking_atom" (ground) if possible.
    """
    if max_depth is None:
        max_depth = config.DEFAULT_MAX_DEPTH

    facts, rules = parse_kb(kb)

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

        if depth >= max_depth:
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

        for num, fact in facts:
            if check_exact_match(current, fact):
                if config.VERBOSE:
                    print(f"  ✓ Fact {num} matches exactly!")

                new_path = path + [f"Fact {num}"]
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

        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True

            if config.VERBOSE:
                print(f"  ✓ Fact {num}: {fact}")
                print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)
            new_path = path + [f"Fact {num}"]

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

        matching_rules = find_matching_rules_only(current, rules)
        if config.VERBOSE and matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num != rule_num:
                    continue
                subgoals = get_subgoals(current, head, body)
                if subgoals:
                    if config.VERBOSE:
                        print(f"  Rule {num}: → {subgoals}")
                    progress = True
                    all_goals = subgoals + remaining
                    queue.append((all_goals[0], all_goals[1:], path + [f"Rule {num}"], depth + 1))
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


def bfs_prolog_metro_soft(
    goal: str,
    kb: str,
    soft_kb,
    max_depth: Optional[int] = None,
    max_soft: Optional[int] = None,
    max_solutions: Optional[int] = None,
):
    """Soft BFS with confidence-first priority + penalty."""
    import heapq
    from itertools import count

    if max_depth is None:
        max_depth = config.DEFAULT_MAX_DEPTH
    if max_solutions is None:
        max_solutions = config.SOFT_BFS_MAX_SOLUTIONS

    hard_facts, hard_rules = parse_kb(kb)

    soft_facts = soft_kb.get("facts", [])
    soft_rules = soft_kb.get("rules", [])
    soft_rules_for_match = [(num, head, body_str) for (num, head, body_str, conf, penalty) in soft_rules]

    if config.VERBOSE:
        print(f"\n[SOFT BFS - BEST-PROOF (conf-priority + penalty)] Goal: {goal}")
        print("-" * 40)

    pq = []
    tie = count()

    def push_state(current, remaining, path, depth, soft_cost, min_conf, penalty_sum):
        if depth > max_depth:
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
        if len(solutions) > max_solutions:
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

        for num, fact in hard_facts:
            if check_exact_match(current, fact):
                if config.VERBOSE:
                    print(f"  ✓ Hard Fact {num} matches exactly: {fact}")
                new_path = path + [f"HardFact {num}"]

                if not remaining:
                    res = make_success_result(new_path, soft_cost, min_conf, depth + 1, penalty_sum)
                    maybe_add_solution(res, soft_cost, min_conf, penalty_sum)
                    break

                push_state(remaining[0], remaining[1:], new_path, depth + 1, soft_cost, min_conf, penalty_sum)
                break

        for num, fact in hard_facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            if config.VERBOSE:
                print(f"  ✓ Hard Fact {num} unifies: {fact}")
                print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)
            new_path = path + [f"HardFact {num}"]

            if not instantiated:
                res = make_success_result(new_path, soft_cost, min_conf, depth + 1, penalty_sum)
                maybe_add_solution(res, soft_cost, min_conf, penalty_sum)
                continue

            push_state(instantiated[0], instantiated[1:], new_path, depth + 1, soft_cost, min_conf, penalty_sum)

        matching_hard_rules = find_matching_rules_only(current, hard_rules)
        if config.VERBOSE and matching_hard_rules:
            print(f"  Matching hard rules: {matching_hard_rules}")

        for rule_num in matching_hard_rules:
            for num, head, body in hard_rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body)
                if not subgoals:
                    continue

                if config.VERBOSE:
                    print(f"  Hard Rule {num}: {head} :- {body}")
                    print(f"    → {subgoals}")

                all_goals = subgoals + remaining
                push_state(all_goals[0], all_goals[1:], path + [f"HardRule {num}"], depth + 1,
                           soft_cost, min_conf, penalty_sum)
                break

        for s_num, s_atom, s_conf, s_penalty in soft_facts:
            if max_soft is not None and soft_cost >= max_soft:
                break

            bindings = unify_with_fact(current, s_atom)
            if bindings is None:
                continue

            new_soft_cost = soft_cost + 1
            new_min_conf = min(min_conf, s_conf)
            new_penalty_sum = penalty_sum + float(s_penalty)

            if config.VERBOSE:
                print(f"  ✓ Soft Fact {s_num} unifies: {s_atom}")
                print(f"    Bindings: {bindings}, conf={s_conf:.3f}, penalty={float(s_penalty):.3f}")
                print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}, new penalty: {new_penalty_sum:.3f}")

            instantiated = apply_bindings(remaining, bindings)
            new_path = path + [f"SoftFact {s_num} (conf={s_conf:.3f},pen={float(s_penalty):.3f})"]

            if not instantiated:
                res = make_success_result(new_path, new_soft_cost, new_min_conf, depth + 1, new_penalty_sum)
                maybe_add_solution(res, new_soft_cost, new_min_conf, new_penalty_sum)
                continue

            push_state(instantiated[0], instantiated[1:], new_path, depth + 1,
                       new_soft_cost, new_min_conf, new_penalty_sum)

        matching_soft_rules = find_matching_rules_only(current, soft_rules_for_match)
        if config.VERBOSE and matching_soft_rules:
            print(f"  Matching soft rules: {matching_soft_rules}")

        for rule_num in matching_soft_rules:
            if max_soft is not None and soft_cost >= max_soft:
                break

            for s_num, s_head, s_body_str, s_conf, s_penalty in soft_rules:
                if s_num != rule_num:
                    continue

                subgoals = get_subgoals(current, s_head, s_body_str)
                if not subgoals:
                    continue

                new_soft_cost = soft_cost + 1
                new_min_conf = min(min_conf, s_conf)
                new_penalty_sum = penalty_sum + float(s_penalty)

                if config.VERBOSE:
                    print(f"  Soft Rule {s_num}: {s_head} :- {s_body_str}")
                    print(f"    → {subgoals}, conf={s_conf:.3f}, penalty={float(s_penalty):.3f}")
                    print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}, new penalty: {new_penalty_sum:.3f}")

                all_goals = subgoals + remaining
                new_path = path + [f"SoftRule {s_num} (conf={s_conf:.3f},pen={float(s_penalty):.3f})"]

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
