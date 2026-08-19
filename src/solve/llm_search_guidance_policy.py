"""LLM-backed search guidance policy for DFS proof search."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Set, Tuple

import config
from log_setup import APP_LOGGER, ERROR_LOGGER, get_logger
from logic.logic import AtomicFormula, Const
from prolog.formula_parsing import split_head_and_body
from prolog.knowledge_base import SoftKnowledgeBase
from prolog.prolog_command import SoftFact, SoftRule
from prolog_llm.llm import LLMInterface
from prolog_llm.prolog_utils import extract_first_json
from solve.goalnode import GoalNode
from solve.prompt_buidler import PromptBuilder, LLMSearchGuidancePromptContext
from solve.search_guidance_policy import SearchGuidancePolicy

_err_log = get_logger(ERROR_LOGGER)
_prompt_log = get_logger(APP_LOGGER)


def coerce_confidence(item) -> Optional[float]:
    """Confidence of one proposed-clause dict, clamped to [0, 1].

    ``None`` means the item is unusable (not a dict, or a non-numeric
    confidence) and should be skipped.
    """
    if not isinstance(item, dict):
        return None
    try:
        confidence = float(item.get("confidence", 0.0))
    except Exception:
        _err_log.exception("coerce_confidence: non-numeric confidence in proposed clause %r; skipping", item)
        return None
    return max(0.0, min(1.0, confidence))


def parse_clause_item(item) -> Optional[Tuple[str, Optional[str], float]]:
    """One proposed clause as ``(head_or_atom, body_or_None, confidence)``.

    ``None`` means the item could not be used at all. A ``body_or_None`` of
    ``None`` marks a fact; anything else is a rule body.
    """
    confidence = coerce_confidence(item)
    if confidence is None:
        return None
    head_or_atom, maybe_body = split_head_and_body(item.get("clause", ""))
    if head_or_atom is None and maybe_body is None:
        return None
    return head_or_atom, maybe_body, confidence


def apply_clauses(clauses: list, soft_kb: SoftKnowledgeBase,
                  allow_soft_rules: bool = False) -> Tuple[SoftKnowledgeBase, List[SoftFact], bool]:
    """Add LLM-/baseline-proposed clauses to a copy of ``soft_kb``.

    Each ``item`` is a dict ``{"clause": "<prolog clause>", "confidence": <float>}``.
    Facts go through ``SoftKnowledgeBase.upsert_soft_fact`` (canonical-atom
    matching); rules are added only when ``allow_soft_rules`` is True. Returns
    ``(new_kb, new_soft_facts, extended)``.

    Note ``extended`` reflects the FACTS only — with ``allow_soft_rules=True`` a
    proposal consisting solely of rules returns ``extended=False``. Harmless
    today (``SolverConfig.allow_soft_rules`` is False), but a caller that turns
    rules on should test the rule list itself.

    Shared by ``LLMSearchGuidancePolicy`` (PLLM), the experiment baselines, and
    the legacy ``solve_bfs.LLMExtensionStrategy``, so hypotheses are parsed and
    inserted identically everywhere.
    """
    new_kb = soft_kb.copy()
    next_num = max([0] + [fact.num for fact in new_kb.facts] + [rule.num for rule in new_kb.rules]) + 1
    new_soft_facts: List[SoftFact] = []

    for item in clauses:
        parsed = parse_clause_item(item)
        if parsed is None:
            continue
        head_or_atom, maybe_body, confidence = parsed

        if maybe_body is None:
            fact, appended = new_kb.upsert_soft_fact(next_num, head_or_atom, confidence)
            if fact is not None:
                new_soft_facts.append(fact)
            if appended:
                next_num += 1
        elif allow_soft_rules:
            new_kb.rules.append(SoftRule(next_num, head_or_atom, maybe_body, confidence))
            next_num += 1

    return new_kb, new_soft_facts, len(new_soft_facts) != 0


def _fact_constants(formula: AtomicFormula) -> List[str]:
    return [arg.name for arg in formula.args if isinstance(arg, Const)]


def select_hard_facts(hard_facts: List[SoftFact], seed_constants: Set[str],
                      limit: int, hops: int) -> Tuple[List[SoftFact], int]:
    """Pick the hard facts worth putting in a prompt, nearest to the goal first.

    Every hard fact used to go into every prompt in full.  That is fine for a
    20-fact KB, but the London benchmark's largest tier has 612 of them, which
    would be repeated in every single LLM call - slow, expensive, and over the
    context window of a local model.

    Selection is a goal-relevant subgraph: constants are treated as nodes, two
    constants are adjacent when they co-occur in a hard fact, and facts are kept
    nearest-first from the constants in the goal and in the open subgoals.

    ``hops`` and ``limit`` are independent caps: the radius bounds relevance,
    the limit bounds size, and whichever bites first wins.  The radius is
    deliberately tight (default 3).  At ``extend_on_init`` the only seeds are
    the goal's own constants, so the window is small; as DFS descends, the open
    subgoals become the seeds and the window SLIDES to wherever the search
    actually is - which is where a missing edge has to be proposed.

    Returns ``(kept, n_withheld)``.  A KB at or under ``limit`` is returned
    untouched, so existing small-KB results stay bit-for-bit reproducible.
    """
    if len(hard_facts) <= limit:
        return hard_facts, 0

    adjacency: Dict[str, Set[str]] = {}
    fact_constants: List[List[str]] = []
    for fact in hard_facts:
        constants = _fact_constants(fact.formula)
        fact_constants.append(constants)
        for c in constants:
            adjacency.setdefault(c, set()).update(x for x in constants if x != c)

    # BFS out from the seeds over the constant co-occurrence graph, to `hops`.
    distance: Dict[str, int] = {c: 0 for c in seed_constants if c in adjacency}
    frontier = list(distance)
    for step in range(1, hops + 1):
        nxt: List[str] = []
        for c in frontier:
            for n in adjacency.get(c, ()):
                if n not in distance:
                    distance[n] = step
                    nxt.append(n)
        frontier = nxt
        if not frontier:
            break

    scored: List[Tuple[int, int, SoftFact]] = []
    for i, (fact, constants) in enumerate(zip(hard_facts, fact_constants)):
        if not constants:
            scored.append((0, i, fact))
            continue
        distances = [distance.get(c) for c in constants]
        if any(d is None for d in distances):
            continue  # outside the goal-relevant subgraph
        scored.append((max(distances), i, fact))

    # Nearest first, so if `limit` bites it is the far edge of the ball that
    # goes, never the neighbourhood the search is working in.
    scored.sort(key=lambda t: (t[0], t[1]))
    kept = [fact for _, _, fact in scored[:limit]]
    return kept, len(hard_facts) - len(kept)


class LLMSearchGuidancePolicy(SearchGuidancePolicy):
    """Guide DFS ordering and backtracking by querying an LLM."""

    def __init__(self, llm_search_guidance: PromptBuilder, llm: Optional[LLMInterface] = None,
                 solver_cfg: Optional["SolverConfig"] = None):
        from cfg import SolverConfig
        self.llm = llm or LLMInterface()
        self.cfg = solver_cfg or SolverConfig()
        self.max_hypotheses = self.cfg.max_hypotheses
        self.allow_soft_rules = self.cfg.allow_soft_rules
        self.verify_hypotheses = config.VERIFY_HYPOTHESES
        self.llm_search_guidance = llm_search_guidance
        self._backtrack_called_signatures: set = set()
        self._order_cache: Dict[tuple, List[tuple]] = {}

    def _hard_fact_lines(self, soft_kb: SoftKnowledgeBase,
                         goal: Optional[AtomicFormula] = None,
                         goal_node: Optional[GoalNode] = None) -> List[str]:
        hard = [fact for fact in soft_kb.facts if fact.confidence == 1.0]

        seeds: Set[str] = set()
        if goal is not None:
            seeds.update(_fact_constants(goal))
        if goal_node is not None:
            for formula in goal_node.unresolved_formulas():
                seeds.update(_fact_constants(formula))

        kept, withheld = select_hard_facts(
            hard, seeds, config.HARD_PROMPT_FACT_LIMIT, config.HARD_PROMPT_FACT_HOPS)
        if withheld:
            # Never truncate silently: a prompt that quietly lost half the KB
            # would look like a model failure rather than a harness decision.
            _prompt_log.info(
                "hard facts in prompt: %d of %d (%d withheld; goal-relevant subgraph "
                "within %d hops of %s, limit %d)",
                len(kept), len(hard), withheld, config.HARD_PROMPT_FACT_HOPS,
                sorted(seeds) or "(no seed constants)", config.HARD_PROMPT_FACT_LIMIT,
            )
        return [f"- {fact.atom}." for fact in kept]

    def _hard_rule_lines(self, soft_kb: SoftKnowledgeBase) -> List[str]:
        return [f"- {rule.head} :- {rule.body}." for rule in soft_kb.rules if rule.confidence == 1.0]

    def _soft_fact_lines(self, soft_kb: SoftKnowledgeBase, min_confidence: float) -> List[str]:
        return [
            f"- {fact.atom}. [conf={fact.confidence:.3f}]"
            for fact in soft_kb.facts if min_confidence <= fact.confidence < 1.0
        ]

    def _soft_rule_lines(self, soft_kb: SoftKnowledgeBase, min_confidence: float) -> List[str]:
        return [
            f"- {rule.head} :- {rule.body}. [conf={rule.confidence:.3f}]"
            for rule in soft_kb.rules if min_confidence <= rule.confidence < 1.0
        ]

    def _prompt_context(self, goal: AtomicFormula, soft_kb: SoftKnowledgeBase,
                        min_confidence: float, goal_node: Optional[GoalNode] = None) -> LLMSearchGuidancePromptContext:
        current_goal_lines = (
            [f"- {formula}" for formula in goal_node.unresolved_formulas()]
            if goal_node is not None else [f"- {goal}"]
        )
        return LLMSearchGuidancePromptContext(
            goal=str(goal),
            current_goal_lines=current_goal_lines,
            hard_fact_lines=self._hard_fact_lines(soft_kb, goal, goal_node),
            hard_rule_lines=self._hard_rule_lines(soft_kb),
            soft_fact_lines=self._soft_fact_lines(soft_kb, min_confidence)[: config.HYP_PROMPT_FACT_LIMIT],
            soft_rule_lines=self._soft_rule_lines(soft_kb, min_confidence)[: config.HYP_PROMPT_FACT_LIMIT],
            predicate_comments=dict(sorted(soft_kb.predicate_comments.items())),
            min_confidence=min_confidence,
        )

    def _order_cache_key(self, current_goal_node: GoalNode, goal_nodes: List[GoalNode],
                         min_confidence: float) -> tuple:
        qualifying_sigs = tuple(sorted(
            node.signature() for node in goal_nodes if node.confidence >= min_confidence
        ))
        return current_goal_node.signature(), qualifying_sigs

    def _apply_order_cache(self, cache_key: tuple, goal_nodes: List[GoalNode]) -> Optional[List[GoalNode]]:
        if cache_key not in self._order_cache:
            return None
        remaining = list(goal_nodes)
        result: List[GoalNode] = []
        for sig in self._order_cache[cache_key]:
            for i, node in enumerate(remaining):
                if node.signature() == sig:
                    result.append(remaining.pop(i))
                    break
        result.extend(remaining)
        return result

    def order_goals(self, goal: AtomicFormula, soft_kb: SoftKnowledgeBase, current_goal_node: GoalNode,
                    min_confidence: float, goal_nodes: List[GoalNode]) -> List[GoalNode]:
        if len(goal_nodes) <= 1:
            return goal_nodes

        cache_key = self._order_cache_key(current_goal_node, goal_nodes, min_confidence)
        cached = self._apply_order_cache(cache_key, goal_nodes)
        if cached is not None:
            return cached

        context = self._prompt_context(goal, soft_kb, min_confidence, current_goal_node)
        prompt = self.llm_search_guidance.order_prompt(context, goal_nodes)
        raw = self.llm.ask_with_retry(prompt, repair_schema=self.llm_search_guidance.order_schema())

        try:
            data = json.loads(extract_first_json(raw))
            order = data.get("order", [])
        except Exception:
            _err_log.exception("order_goals: failed to parse LLM ordering response; keeping original order. raw=%r", raw)
            return goal_nodes

        if not isinstance(order, list):
            return goal_nodes

        ordered_goal_nodes: List[GoalNode] = []
        used_indexes = set()

        for idx in order:
            if not isinstance(idx, int) or idx < 0 or idx >= len(goal_nodes) or idx in used_indexes:
                continue
            ordered_goal_nodes.append(goal_nodes[idx])
            used_indexes.add(idx)

        for idx, node in enumerate(goal_nodes):
            if idx not in used_indexes:
                ordered_goal_nodes.append(node)

        self._order_cache[cache_key] = [node.signature() for node in ordered_goal_nodes]
        return ordered_goal_nodes

    def _verify_clauses(self, clauses: list, goal: AtomicFormula, soft_kb: SoftKnowledgeBase,
                        min_confidence: float) -> list:
        """Second LLM pass: keep only clauses judged plausible (not shortcuts).

        Fail-open: on any parse error (or when disabled) the clauses are returned
        unchanged, mirroring the defensive pattern in ``order_goals``.
        """
        if not self.verify_hypotheses or not clauses:
            return clauses

        context = self._prompt_context(goal, soft_kb, min_confidence)
        prompt = self.llm_search_guidance.verify_prompt(
            context, [item.get("clause", "") for item in clauses]
        )
        raw = self.llm.ask_with_retry(prompt, repair_schema=self.llm_search_guidance.verify_schema())

        try:
            data = json.loads(extract_first_json(raw))
            verdicts = data.get("verdicts", [])
            kept = {
                v.get("clause", "") for v in verdicts
                if isinstance(v, dict) and v.get("keep") is True
            }
        except Exception:
            _err_log.exception("_verify_clauses: failed to parse LLM verdicts; keeping all clauses. raw=%r", raw)
            return clauses

        return [item for item in clauses if item.get("clause", "") in kept]

    def _apply_clauses(self, clauses: list, soft_kb: SoftKnowledgeBase) -> Tuple[SoftKnowledgeBase, List[SoftFact], bool]:
        return apply_clauses(clauses, soft_kb, allow_soft_rules=self.allow_soft_rules)

    def extend_on_backtrack(self, goal: AtomicFormula, goal_node: GoalNode, min_confidence: float,
                            soft_kb: SoftKnowledgeBase) -> Tuple[SoftKnowledgeBase, List[SoftFact], bool]:
        signature = goal_node.signature()
        if signature in self._backtrack_called_signatures:
            return soft_kb, [], False
        self._backtrack_called_signatures.add(signature)

        context = self._prompt_context(goal, soft_kb, min_confidence, goal_node)
        prompt = self.llm_search_guidance.backtrack_prompt(context)
        raw = self.llm.ask_with_retry(prompt, repair_schema=self.llm_search_guidance.backtrack_schema())

        try:
            data = json.loads(extract_first_json(raw))
        except Exception:
            _err_log.exception("extend_on_backtrack: failed to parse LLM clauses response. raw=%r", raw)
            return soft_kb, [], False

        clauses = data.get("clauses", [])
        if not isinstance(clauses, list):
            return soft_kb, [], False

        clauses = self._verify_clauses(clauses, goal, soft_kb, min_confidence)
        return self._apply_clauses(clauses, soft_kb)

    def estimate_depth(self, goal: AtomicFormula, soft_kb: SoftKnowledgeBase) -> int:
        context = self._prompt_context(goal, soft_kb, 1.0)
        prompt = self.llm_search_guidance.estimate_depth_prompt(context)
        raw = self.llm.ask_with_retry(prompt, repair_schema=self.llm_search_guidance.estimate_depth_schema())
        try:
            data = json.loads(extract_first_json(raw))
            return max(1, int(data.get("depth", config.DEFAULT_MAX_DEPTH)))
        except Exception:
            _err_log.exception("estimate_depth: failed to parse LLM depth response; using default. raw=%r", raw)
            return config.DEFAULT_MAX_DEPTH

    def extend_on_init(self, goal: AtomicFormula, min_confidence: float,
                       soft_kb: SoftKnowledgeBase) -> Tuple[SoftKnowledgeBase, List[SoftFact], bool]:
        context = self._prompt_context(goal, soft_kb, min_confidence)
        prompt = self.llm_search_guidance.extend_on_init_prompt(context)
        raw = self.llm.ask_with_retry(prompt, repair_schema=self.llm_search_guidance.backtrack_schema())

        try:
            data = json.loads(extract_first_json(raw))
        except Exception:
            _err_log.exception("extend_on_init: failed to parse LLM clauses response. raw=%r", raw)
            return soft_kb, [], False

        clauses = data.get("clauses", [])
        if not isinstance(clauses, list):
            return soft_kb, [], False

        clauses = self._verify_clauses(clauses, goal, soft_kb, min_confidence)
        return self._apply_clauses(clauses, soft_kb)