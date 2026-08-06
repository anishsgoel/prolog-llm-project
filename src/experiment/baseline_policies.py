"""Policies and KB helpers used by the experiment baselines.

Kept out of ``src/solve`` so the core solver package needs no changes: the
random proposer is a drop-in :class:`SearchGuidancePolicy`, and the ablation
policies are thin subclasses of :class:`TrackedLLMSearchGuidancePolicy` that
flip a single behaviour off.

Also provides the candidate ground-atom enumeration the symbolic baselines
(``minimal_alp``) and the random baseline (``random_edge``) need, which the
codebase does not otherwise expose.
"""

from __future__ import annotations

import random
from itertools import product
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import config
from logic.logic import AtomicFormula, BinaryFormula, Const, Pred
from prolog.knowledge_base import KnowledgeBase, SoftKnowledgeBase
from prolog.prolog_command import SoftFact
from solve.goalnode import GoalNode
from solve.llm_search_guidance_policy import apply_clauses
from solve.search_guidance_policy import SearchGuidancePolicy

from experiment.tracking import TrackedLLMSearchGuidancePolicy


# ---------------------------------------------------------------------------
# Constant / candidate-atom enumeration
# ---------------------------------------------------------------------------

def parse_pred_arity(spec: str) -> Tuple[str, int]:
    """Split a ``"connected/2"`` style abducible spec into ``("connected", 2)``."""
    name, _, arity = spec.rpartition("/")
    return name.strip(), int(arity)


def _formula_consts(formula) -> Set[Const]:
    """Constants of a formula, safe for rule bodies.

    ``AtomicFormula`` (and other ``QFFormula``) expose ``.consts()``, but a
    multi-atom rule body is a :class:`Conjunction` (a ``BinaryFormula`` /
    ``CompoundFormula``) which does NOT implement ``.consts()`` — calling it
    blindly raises ``AttributeError``. We recurse over binary formulas and only
    call ``.consts()`` on leaves that support it.
    """
    if formula is None:
        return set()
    if isinstance(formula, BinaryFormula):
        return _formula_consts(formula.left_formula) | _formula_consts(formula.right_formula)
    consts = getattr(formula, "consts", None)
    if callable(consts):
        try:
            return set(consts())
        except Exception:
            return set()
    return set()


def kb_constants(kb: KnowledgeBase) -> List[Const]:
    """All constants appearing in the KB's facts and rules, sorted by name.

    In practice every constant lives in the facts (rules are variable-only
    schemas), but rule heads/bodies are scanned too for completeness.
    """
    consts: Set[Const] = set()
    for fact in kb.facts:
        consts |= set(fact.formula.consts())
    for rule in kb.rules:
        consts |= _formula_consts(rule.head_formula)
        consts |= _formula_consts(rule.body_formula)
    return sorted(consts, key=lambda c: c.name)


def predicate_signatures(kb: KnowledgeBase) -> Set[Tuple[str, int]]:
    """Set of ``(predicate_name, arity)`` present among the KB's facts."""
    return {(f.formula.pred.name, f.formula.pred.arity) for f in kb.facts}


def candidate_atoms(kb: KnowledgeBase, propose_facts: Sequence[str]) -> List[str]:
    """Canonical atom strings for every abducible predicate over the KB constants.

    For each ``pred/arity`` in ``propose_facts`` we form ``pred`` applied to
    every tuple in ``product(sorted_constants, repeat=arity)`` and keep the ones
    NOT already present in ``kb`` (compared canonically via
    ``AtomicFormula.__str__``). The returned strings are canonical so that
    ``SoftKnowledgeBase.add_soft_fact`` (which matches by exact atom string)
    never creates a phantom duplicate of an existing fact.

    Note: the pool is restricted to constants already in the KB. There is no
    vocabulary of "new" constants to enumerate, so on problems with
    ``allow_new_constants=True`` a symbolic baseline can only rediscover facts
    over the existing constants.
    """
    consts = kb_constants(kb)
    existing = {str(f.formula) for f in kb.facts}
    seen: Set[str] = set()
    out: List[str] = []
    for spec in propose_facts:
        name, arity = parse_pred_arity(spec)
        pred = Pred(name, arity)
        for combo in product(consts, repeat=arity):
            atom = str(pred(*combo))
            if atom in existing or atom in seen:
                continue
            seen.add(atom)
            out.append(atom)
    return out


def add_candidate_facts(soft_kb: SoftKnowledgeBase, atoms: Iterable[str],
                        confidence: float) -> Tuple[SoftKnowledgeBase, List[SoftFact], bool]:
    """Add each canonical ``atom`` to a copy of ``soft_kb`` at ``confidence``.

    Thin adapter that formats plain atom strings as clause dicts and delegates
    to :func:`solve.llm_search_guidance_policy.apply_clauses`, so the baselines
    reuse the exact fact-insertion logic PLLM uses (no reimplementation).
    """
    clauses = [{"clause": atom, "confidence": confidence} for atom in atoms]
    return apply_clauses(clauses, soft_kb, allow_soft_rules=False)


# ---------------------------------------------------------------------------
# random_edge policy
# ---------------------------------------------------------------------------

class RandomFactProposerPolicy(SearchGuidancePolicy):
    """A ``SearchGuidancePolicy`` whose "LLM" proposes RANDOM candidate facts.

    Drop-in replacement for the LLM policy inside ``DFSMetaSolver`` so the
    search machinery is identical to PLLM; only the source of hypotheses
    changes. Isolates the LLM's world knowledge from the search scaffold.

    Ordering is identity and the depth estimate is the configured default (no
    "LLM" involvement). On each init/backtrack extension it hands out up to
    ``max_hypotheses`` fresh candidate atoms (drawn from a seeded shuffle of the
    candidate pool) at a fixed confidence, and reports "no extension" once the
    pool is exhausted.
    """

    def __init__(self, candidates: Sequence[str], max_hypotheses: int,
                 seed: int = 0, confidence: float = 0.5):
        self.max_hypotheses = max(1, int(max_hypotheses))
        self.confidence = confidence
        self._pool = list(candidates)
        random.Random(seed).shuffle(self._pool)
        self._cursor = 0

    def order_goals(self, goal: AtomicFormula, soft_kb: SoftKnowledgeBase,
                    current_goal_node: GoalNode, min_confidence: float,
                    goal_nodes: List[GoalNode]) -> List[GoalNode]:
        return goal_nodes

    def estimate_depth(self, goal: AtomicFormula, soft_kb: SoftKnowledgeBase) -> int:
        return config.DEFAULT_MAX_DEPTH

    def _propose(self, soft_kb: SoftKnowledgeBase) -> Tuple[SoftKnowledgeBase, List[SoftFact], bool]:
        existing = {str(f.formula) for f in soft_kb.facts}
        chosen: List[str] = []
        while self._cursor < len(self._pool) and len(chosen) < self.max_hypotheses:
            cand = self._pool[self._cursor]
            self._cursor += 1
            if cand not in existing:
                chosen.append(cand)
        if not chosen:
            return soft_kb, [], False
        return add_candidate_facts(soft_kb, chosen, self.confidence)

    def extend_on_init(self, goal: AtomicFormula, min_confidence: float,
                       soft_kb: SoftKnowledgeBase) -> Tuple[SoftKnowledgeBase, List[SoftFact], bool]:
        return self._propose(soft_kb)

    def extend_on_backtrack(self, goal: AtomicFormula, goal_node: GoalNode,
                            min_confidence: float,
                            soft_kb: SoftKnowledgeBase) -> Tuple[SoftKnowledgeBase, List[SoftFact], bool]:
        return self._propose(soft_kb)


# ---------------------------------------------------------------------------
# PLLM ablation policies (toggle a single behaviour off)
# ---------------------------------------------------------------------------

class NoOrderPolicy(TrackedLLMSearchGuidancePolicy):
    """PLLM with the LLM goal-ordering step disabled (keeps DFS input order)."""

    def order_goals(self, goal, soft_kb, current_goal_node, min_confidence, goal_nodes):
        return goal_nodes


class FixedDepthPolicy(TrackedLLMSearchGuidancePolicy):
    """PLLM with the LLM depth estimate replaced by a constant (no depth call)."""

    def estimate_depth(self, goal, soft_kb) -> int:
        self.depth_estimate = config.DEFAULT_MAX_DEPTH
        return config.DEFAULT_MAX_DEPTH


class InitOnlyPolicy(TrackedLLMSearchGuidancePolicy):
    """PLLM that hypothesises only once (before search); never on backtrack."""

    def extend_on_backtrack(self, goal, goal_node, min_confidence, soft_kb):
        return soft_kb, [], False


class BacktrackOnlyPolicy(TrackedLLMSearchGuidancePolicy):
    """PLLM that hypothesises only during backtracking; not before search."""

    def extend_on_init(self, goal, min_confidence, soft_kb):
        return soft_kb, [], False