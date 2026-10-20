"""Baselines and PLLM ablations, all emitting the same RunMetrics.

Every runner has the signature ``run_<name>(config_name, model=None, **kwargs)``
and ends by calling :func:`experiment.tracking.compute_metrics`, so the whole
comparison table is computed identically and is directly comparable to PLLM.

Hypotheses are parsed and inserted through the SAME helpers PLLM uses:
:func:`solve.llm_search_guidance_policy.apply_clauses` (clause dict -> soft KB)
and :func:`prolog_llm.prolog_utils.extract_first_json`. Nothing re-implements
that logic here.

Registries
----------
``METHODS``   – full methods (``pllm`` plus the standalone baselines).
``ABLATIONS`` – PLLM with a single knob flipped.

Use :func:`run_method` as the single dispatch entry point; ``batch_run`` and
``timed_run`` call it and forward ``method`` / ``ablation`` / ``run_index``
across the process-spawn boundary (module globals do NOT cross that boundary,
so the choice must travel as an argument).

Baselines
---------
no_abduction  – plain solver on the ablated KB (floor; must fail).
random_edge   – DFS meta-solver whose "LLM" proposes RANDOM candidate facts.
minimal_alp   – brute-force minimum-cardinality symbolic abduction (no LLM).
llm_oneshot   – one LLM call proposes facts; symbolic solver verifies.
llm_qa        – one LLM call answers the goal true/false (no proof).
logic_lm      – LLM propose -> symbolic solve -> self-refine loop (literature).
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from typing import Any, Dict, List, Optional

import config
from experiment.baseline_policies import (
    BacktrackOnlyPolicy,
    FixedDepthPolicy,
    InitOnlyPolicy,
    NoOrderPolicy,
    NoSoftFactsInPromptPolicy,
    RandomFactProposerPolicy,
    UnifyTargetPolicy,
    add_candidate_facts,
    candidate_atoms,
)
from experiment.prompts import PrologPromptBuilder
from experiment.tracking import (
    GROUND_TRUTH_TIMEOUT_S,
    RunMetrics,
    TrackedLLMInterface,
    TrackedLLMSearchGuidancePolicy,
    build_problem,
    compute_metrics,
)
from prolog.knowledge_base import SoftKnowledgeBase
from prolog_llm.prolog_utils import extract_first_json
from solve import DFSMetaSolver, LLMSearchGuidancePolicy
from solve.dfssolver import DFSSolver
from solve.llm_search_guidance_policy import apply_clauses
from solve.search_guidance_policy import TrivialSearchGuidancePolicy

# Confidence assigned to the hypotheses proposed by the non-LLM baselines. Must
# be < 1.0 so the facts count as hypotheses (soft facts) in compute_metrics.
_HYP_CONFIDENCE = 0.5


def _prove(kb, goal, max_depth: int) -> Dict[str, Any]:
    """Verify a (soft) KB proves ``goal`` using the SAME prover PLLM uses.

    Uses ``DFSSolver`` with the no-op guidance policy and ``allow_extension=False``
    (a pure deduction check — no fact injection). This is the honest "does this
    KB prove the goal?" oracle for the symbolic/LLM baselines. We deliberately do
    NOT use the priority-queue ``Solver`` here: with ``only_grounded=False`` rule
    unification it blows up combinatorially at the ceiling depth on the recursive
    rules (tens of seconds per failing probe), which ``minimal_alp`` would call
    thousands of times. Returns ``{success, proof, confidence}``.
    """
    solver = DFSSolver(kb, max_depth=max_depth,
                       search_guidance_policy=TrivialSearchGuidancePolicy())
    return solver.solve(goal, min_confidence=0.0, allow_extension=False)


def _empty_result(soft_kb: SoftKnowledgeBase, max_depth: int) -> Dict[str, Any]:
    """A DFSMetaSolver-shaped result for a failed / no-proof run."""
    return {
        "success": False, "proof": None, "confidence": 0.0,
        "soft_kb": soft_kb, "max_depth": max_depth,
        "min_confidence": 0.0, "attempts": [],
    }


def _ground_truth_note(gt: Optional[tuple]) -> None:
    if gt is None:
        print(f"  ground-truth search exceeded {GROUND_TRUTH_TIMEOUT_S}s -> recording NA")


def _clauses_from_raw(raw: str) -> list:
    """Pull the clause list out of an LLM JSON response, fail-open to ``[]``.

    Accepts either the ``{"facts": [...]}`` envelope used by the baseline
    templates or the ``{"clauses": [...]}`` envelope used by the DFS templates;
    the per-clause parsing/insertion is then done by ``apply_clauses``.
    """
    try:
        data = json.loads(extract_first_json(raw))
    except Exception:
        return []
    for key in ("facts", "clauses"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


# ---------------------------------------------------------------------------
# pllm
# ---------------------------------------------------------------------------

def run_pllm(config_name: str, model: Optional[str] = None, **kwargs) -> RunMetrics:
    """The system under study: LLM-guided DFS meta-solver. Delegates to
    ``run_tracked`` so PLLM numbers are exactly the tracked-run numbers."""
    from experiment.tracking import run_tracked
    return run_tracked(config_name, model=model)


# ---------------------------------------------------------------------------
# no_abduction (floor)
# ---------------------------------------------------------------------------

def run_no_abduction(config_name: str, model: Optional[str] = None, **kwargs) -> RunMetrics:
    """Plain solver on the ablated KB, with NO fact injection. Expected to
    fail: if it succeeds, the omitted facts were not load-bearing."""
    problem, kb_text, kb_full, kb_missing, goal_formula, gt = build_problem(config_name, model)
    _ground_truth_note(gt)

    max_depth = problem.solver.max_depth_ceiling
    t0 = time.perf_counter()
    res = _prove(kb_missing, goal_formula, max_depth)
    wall = time.perf_counter() - t0

    if res.get("success"):
        print(f"  [no_abduction] WARNING: {config_name} proved WITHOUT the omitted "
              f"facts (omit_fact_ids={problem.omit_fact_ids} are not load-bearing).")

    result = {
        "success": res.get("success", False),
        "proof": res.get("proof"),
        "confidence": res.get("confidence", 0.0),
        "soft_kb": SoftKnowledgeBase(kb_missing),  # no hypotheses
        "max_depth": max_depth,
        "min_confidence": 0.0,
        "attempts": [],
    }
    return compute_metrics(config_name, config.MODEL, problem, kb_full, kb_missing,
                           result, None, None, gt, wall)


# ---------------------------------------------------------------------------
# minimal_alp (symbolic minimum-cardinality abduction)
# ---------------------------------------------------------------------------

def run_minimal_alp(config_name: str, model: Optional[str] = None, **kwargs) -> RunMetrics:
    """Classical abductive logic programming baseline.

    Finds a MINIMUM-CARDINALITY set Δ of ground abducible facts whose addition
    makes the goal provable, by exhaustive search over increasing |Δ|.

    Symbolic and LLM-free: it closes the proof with whatever facts work, which
    is typically a "shortcut" edge — high success and minimal proofs, but low
    precision/recall against the true model (it has no world knowledge).

    References:
      * Kakas, Kowalski & Toni, "Abductive Logic Programming",
        Journal of Logic and Computation 2(6), 1992.
      * Denecker & Kakas, "Abduction in Logic Programming",
        Computational Logic: Logic Programming and Beyond, LNCS 2407,
        Springer, 2002.
    """
    problem, kb_text, kb_full, kb_missing, goal_formula, gt = build_problem(config_name, model)
    _ground_truth_note(gt)

    max_depth = problem.solver.max_depth_ceiling
    candidates = candidate_atoms(kb_missing, problem.propose_facts)
    # K_MAX: |omitted| + 1, hard-capped at 3 (combinatorics blow up quickly).
    k_max = int(kwargs.get("k_max", min(len(problem.omit_fact_ids) + 1, 3)))
    k_max = max(1, k_max)

    attempts: List[Dict[str, Any]] = []
    found_delta: Optional[tuple] = None
    found_proof = None
    found_conf = 0.0

    t0 = time.perf_counter()
    for k in range(1, k_max + 1):
        for delta in combinations(candidates, k):
            soft_kb, _, _ = add_candidate_facts(SoftKnowledgeBase(kb_missing), delta, _HYP_CONFIDENCE)
            res = _prove(soft_kb, goal_formula, max_depth)
            attempts.append({"k": k, "success": bool(res.get("success"))})
            if res.get("success"):
                found_delta = delta
                found_proof = res.get("proof")
                found_conf = res.get("confidence", 0.0)
                break
        if found_delta is not None:
            break
    wall = time.perf_counter() - t0

    if found_delta is not None:
        soft_kb, _, _ = add_candidate_facts(SoftKnowledgeBase(kb_missing), found_delta, _HYP_CONFIDENCE)
        result = {
            "success": True, "proof": found_proof, "confidence": found_conf,
            "soft_kb": soft_kb, "max_depth": max_depth,
            "min_confidence": 0.0, "attempts": attempts,
        }
    else:
        print(f"  [minimal_alp] no Δ up to cardinality {k_max} proved {config_name} "
              f"({len(candidates)} candidate atoms) -> capped, reporting failure.")
        result = _empty_result(SoftKnowledgeBase(kb_missing), max_depth)
        result["attempts"] = attempts

    return compute_metrics(config_name, config.MODEL, problem, kb_full, kb_missing,
                           result, None, None, gt, wall)


# ---------------------------------------------------------------------------
# random_edge (search scaffold with random hypotheses)
# ---------------------------------------------------------------------------

def run_random_edge(config_name: str, model: Optional[str] = None, **kwargs) -> RunMetrics:
    """DFS meta-solver identical to PLLM, but the hypotheses are RANDOM candidate
    facts instead of LLM proposals. Isolates the search machinery from the LLM's
    knowledge. Uses a seeded RNG (seed = run_index) for reproducibility."""
    problem, kb_text, kb_full, kb_missing, goal_formula, gt = build_problem(config_name, model)
    _ground_truth_note(gt)

    seed = int(kwargs.get("run_index", 0))
    candidates = candidate_atoms(kb_missing, problem.propose_facts)
    policy = RandomFactProposerPolicy(
        candidates, max_hypotheses=problem.solver.max_hypotheses,
        seed=seed, confidence=_HYP_CONFIDENCE,
    )
    meta_solver = DFSMetaSolver(kb_missing, policy, solver_cfg=problem.solver)

    t0 = time.perf_counter()
    result = meta_solver.solve(goal_formula)
    wall = time.perf_counter() - t0

    return compute_metrics(config_name, config.MODEL, problem, kb_full, kb_missing,
                           result, None, policy, gt, wall)


# ---------------------------------------------------------------------------
# llm_oneshot (LLM proposes, symbolic solver verifies)
# ---------------------------------------------------------------------------

def run_llm_oneshot(config_name: str, model: Optional[str] = None, **kwargs) -> RunMetrics:
    """One LLM call proposes the missing ground facts; the SYMBOLIC solver then
    checks whether they actually yield a proof. Tests whether the iterative DFS
    scaffold beats a single prompt."""
    problem, kb_text, kb_full, kb_missing, goal_formula, gt = build_problem(config_name, model)
    _ground_truth_note(gt)

    max_depth = problem.solver.max_depth_ceiling
    tracked_llm = TrackedLLMInterface(model=config.MODEL)
    tracked_llm.set_call_type("formalize")
    builder = PrologPromptBuilder(problem.propose_facts, problem.solver,
                                  allow_new_constants=problem.allow_new_constants)

    t0 = time.perf_counter()
    prompt = builder.oneshot_completion_prompt(problem.goal, kb_missing.to_text(),
                                               kb_missing.predicate_comments)
    raw = tracked_llm.ask_with_retry(prompt, repair_schema='{"facts": []}')
    soft_kb, _, _ = apply_clauses(_clauses_from_raw(raw), SoftKnowledgeBase(kb_missing))
    res = _prove(soft_kb, goal_formula, max_depth)
    wall = time.perf_counter() - t0

    result = {
        "success": res.get("success", False),
        "proof": res.get("proof"),
        "confidence": res.get("confidence", 0.0),
        "soft_kb": soft_kb, "max_depth": max_depth,
        "min_confidence": 0.0, "attempts": [{"formalize": True}],
    }
    return compute_metrics(config_name, config.MODEL, problem, kb_full, kb_missing,
                           result, tracked_llm, None, gt, wall)


# ---------------------------------------------------------------------------
# llm_qa (direct QA, no solver)
# ---------------------------------------------------------------------------

def run_llm_qa(config_name: str, model: Optional[str] = None, **kwargs) -> RunMetrics:
    """One LLM call answers the goal true/false, with NO solver and NO proof.
    Tests whether a solver is needed at all.

    Figure of merit: ``success`` (the predicted boolean) vs ``ground_truth_success``.
    Because there is no proof, valid_success / proof_matches_gt / precision /
    recall / hallucination are all 0 here — that is N/A for QA, not a failure.
    """
    problem, kb_text, kb_full, kb_missing, goal_formula, gt = build_problem(config_name, model)
    _ground_truth_note(gt)

    tracked_llm = TrackedLLMInterface(model=config.MODEL)
    tracked_llm.set_call_type("qa")
    builder = PrologPromptBuilder(problem.propose_facts, problem.solver,
                                  allow_new_constants=problem.allow_new_constants)

    t0 = time.perf_counter()
    prompt = builder.qa_prompt(problem.goal, kb_missing.to_text(), kb_missing.predicate_comments)
    raw = tracked_llm.ask_with_retry(prompt, repair_schema='{"answer": false}')
    try:
        predicted = bool(json.loads(extract_first_json(raw)).get("answer", False))
    except Exception:
        predicted = False
    wall = time.perf_counter() - t0

    result = {
        "success": predicted, "proof": None, "confidence": 0.0,
        "soft_kb": SoftKnowledgeBase(kb_missing),  # no hypotheses
        "max_depth": 0, "min_confidence": 0.0, "attempts": [{"qa": True}],
    }
    return compute_metrics(config_name, config.MODEL, problem, kb_full, kb_missing,
                           result, tracked_llm, None, gt, wall)


# ---------------------------------------------------------------------------
# logic_lm (LLM propose -> symbolic solve -> self-refine)
# ---------------------------------------------------------------------------

def run_logic_lm(config_name: str, model: Optional[str] = None, **kwargs) -> RunMetrics:
    """Logic-LM-style baseline adapted to KB completion.

    Logic-LM (Pan, Albalak, Wang, Wang, "Logic-LM: Empowering Large Language
    Models with Symbolic Solvers for Faithful Logical Reasoning", Findings of
    EMNLP 2023, arXiv:2305.12295) formalises a problem for a deterministic
    symbolic solver, then a self-refinement module rewrites the formalisation
    from the solver's error messages. Here the KB is already symbolic, so we
    adapt only the loop: the LLM proposes missing facts, the existing solver
    verifies, and on failure the solver's open subgoals are fed back for a
    revised proposal, up to ``refine_budget`` (default 3) rounds.

    ``refine_budget=0`` collapses to ``llm_oneshot``, giving a clean
    within-baseline gradient. Formulate/refine calls are counted separately.
    """
    problem, kb_text, kb_full, kb_missing, goal_formula, gt = build_problem(config_name, model)
    _ground_truth_note(gt)

    refine_budget = int(kwargs.get("refine_budget", 3))
    max_depth = problem.solver.max_depth_ceiling
    tracked_llm = TrackedLLMInterface(model=config.MODEL)
    builder = PrologPromptBuilder(problem.propose_facts, problem.solver,
                                  allow_new_constants=problem.allow_new_constants)

    attempts: List[Dict[str, Any]] = []
    # Accumulate proposals across rounds; the soft KB is rebuilt each round from
    # kb_missing + all clauses proposed so far (apply_clauses de-dupes by atom).
    proposed_clauses: List[dict] = []
    t0 = time.perf_counter()

    # --- Round 0: formalise ---
    tracked_llm.set_call_type("formalize")
    prompt = builder.oneshot_completion_prompt(problem.goal, kb_missing.to_text(),
                                               kb_missing.predicate_comments)
    raw = tracked_llm.ask_with_retry(prompt, repair_schema='{"facts": []}')
    proposed_clauses.extend(_clauses_from_raw(raw))
    soft_kb, _, _ = apply_clauses(proposed_clauses, SoftKnowledgeBase(kb_missing))
    res = _prove(soft_kb, goal_formula, max_depth)
    attempts.append({"round": 0, "kind": "formalize", "success": bool(res.get("success"))})

    # --- Rounds 1..refine_budget: self-refine from solver feedback ---
    # The query here is a single atom, so the "open subgoal" the solver is stuck
    # on is the goal itself; that plus the still-insufficient proposals is the
    # feedback handed back for a revised proposal (Logic-LM-style refine loop).
    round_idx = 0
    while not res.get("success") and round_idx < refine_budget:
        round_idx += 1
        open_lines = [str(goal_formula)]
        proposed_lines = [f"{c.get('clause', '')} [conf={c.get('confidence', 0.0)}]"
                          for c in proposed_clauses] or ["(none)"]

        tracked_llm.set_call_type("refine")
        prompt = builder.logic_lm_refine_prompt(
            problem.goal, kb_missing.to_text(), kb_missing.predicate_comments,
            proposed_lines, open_lines,
        )
        raw = tracked_llm.ask_with_retry(prompt, repair_schema='{"facts": []}')
        proposed_clauses.extend(_clauses_from_raw(raw))
        soft_kb, _, _ = apply_clauses(proposed_clauses, SoftKnowledgeBase(kb_missing))
        res = _prove(soft_kb, goal_formula, max_depth)
        attempts.append({"round": round_idx, "kind": "refine", "success": bool(res.get("success"))})

    wall = time.perf_counter() - t0

    result = {
        "success": res.get("success", False),
        "proof": res.get("proof"),
        "confidence": res.get("confidence", 0.0),
        "soft_kb": soft_kb, "max_depth": max_depth,
        "min_confidence": 0.0, "attempts": attempts,
    }
    return compute_metrics(config_name, config.MODEL, problem, kb_full, kb_missing,
                           result, tracked_llm, None, gt, wall)


# ---------------------------------------------------------------------------
# Ablations of PLLM
# ---------------------------------------------------------------------------

_ABLATION_POLICIES = {
    "no_order": NoOrderPolicy,
    "fixed_depth": FixedDepthPolicy,
    "init_only": InitOnlyPolicy,
    "backtrack_only": BacktrackOnlyPolicy,
    "unify_target": UnifyTargetPolicy,
    "no_soft_facts_in_prompt": NoSoftFactsInPromptPolicy,
}


def run_ablation(config_name: str, model: Optional[str] = None,
                 ablation: Optional[str] = None, **kwargs) -> RunMetrics:
    """Run the PLLM pipeline with a single knob flipped.

    Config-flag ablations are applied AFTER build_problem() (which calls
    init_config() and would otherwise reset them). ``verify_on`` must be set
    BEFORE the policy is constructed because LLMSearchGuidancePolicy snapshots
    ``config.VERIFY_HYPOTHESES`` at __init__; ``min_hypotheses`` can be set any
    time before .solve() because DFSMetaSolver reads config.PROOF_OBJECTIVE live.
    """
    problem, kb_text, kb_full, kb_missing, goal_formula, gt = build_problem(config_name, model)
    _ground_truth_note(gt)

    if ablation == "verify_on":
        config.VERIFY_HYPOTHESES = True          # snapshotted -> set before policy ctor
    if ablation == "min_hypotheses":
        config.PROOF_OBJECTIVE = "min_hypotheses"  # read live in DFSMetaSolver.solve

    tracked_llm = TrackedLLMInterface(model=config.MODEL)
    builder = PrologPromptBuilder(problem.propose_facts, problem.solver,
                                  allow_new_constants=problem.allow_new_constants)
    policy_cls = _ABLATION_POLICIES.get(ablation, TrackedLLMSearchGuidancePolicy)
    policy = policy_cls(builder, tracked_llm=tracked_llm, solver_cfg=problem.solver)
    meta_solver = DFSMetaSolver(kb_missing, policy, solver_cfg=problem.solver)

    t0 = time.perf_counter()
    result = meta_solver.solve(goal_formula)
    wall = time.perf_counter() - t0

    return compute_metrics(config_name, config.MODEL, problem, kb_full, kb_missing,
                           result, tracked_llm, policy, gt, wall)


def run_self_consistency(config_name: str, model: Optional[str] = None, **kwargs) -> RunMetrics:
    """Optional ablation: sample ``k`` independent proposal sets and UNION them
    before a single symbolic verification. Reuses the LLM policy's own
    ``extend_on_init`` to generate each proposal set.

    NOTE on temperature: ``LLMInterface.generate`` currently comments out the
    Ollama ``options`` block, so an explicit ``temperature>0`` cannot be forced;
    this relies on the backend's default sampler nondeterminism to make the k
    proposals differ. Default k=3 (documented; override via ``k`` kwarg).
    """
    problem, kb_text, kb_full, kb_missing, goal_formula, gt = build_problem(config_name, model)
    _ground_truth_note(gt)

    k = int(kwargs.get("k", 3))
    max_depth = problem.solver.max_depth_ceiling
    tracked_llm = TrackedLLMInterface(model=config.MODEL)
    builder = PrologPromptBuilder(problem.propose_facts, problem.solver,
                                  allow_new_constants=problem.allow_new_constants)
    policy = LLMSearchGuidancePolicy(builder, llm=tracked_llm, solver_cfg=problem.solver)

    t0 = time.perf_counter()
    union: Dict[str, float] = {}  # canonical atom -> max confidence seen
    for _ in range(k):
        tracked_llm.set_call_type("extend_on_init")
        _, new_facts, _ = policy.extend_on_init(goal_formula, 1.0, SoftKnowledgeBase(kb_missing))
        for sf in new_facts:
            atom = str(sf.formula)
            union[atom] = max(union.get(atom, 0.0), sf.confidence)

    union_clauses = [{"clause": atom, "confidence": conf} for atom, conf in union.items()]
    soft_kb, _, _ = apply_clauses(union_clauses, SoftKnowledgeBase(kb_missing))
    res = _prove(soft_kb, goal_formula, max_depth)
    wall = time.perf_counter() - t0

    result = {
        "success": res.get("success", False),
        "proof": res.get("proof"),
        "confidence": res.get("confidence", 0.0),
        "soft_kb": soft_kb, "max_depth": max_depth,
        "min_confidence": 0.0, "attempts": [{"samples": k}],
    }
    return compute_metrics(config_name, config.MODEL, problem, kb_full, kb_missing,
                           result, tracked_llm, None, gt, wall)


# ---------------------------------------------------------------------------
# Registries + dispatch
# ---------------------------------------------------------------------------

METHODS = {
    "pllm": run_pllm,
    "no_abduction": run_no_abduction,
    "random_edge": run_random_edge,
    "minimal_alp": run_minimal_alp,
    "llm_oneshot": run_llm_oneshot,
    "llm_qa": run_llm_qa,
    "logic_lm": run_logic_lm,
}


def _ablation_runner(name: str):
    def _run(config_name: str, model: Optional[str] = None, **kwargs) -> RunMetrics:
        return run_ablation(config_name, model, ablation=name, **kwargs)
    _run.__name__ = f"run_ablation_{name}"
    return _run


ABLATIONS = {
    "verify_on": _ablation_runner("verify_on"),
    "min_hypotheses": _ablation_runner("min_hypotheses"),
    "no_order": _ablation_runner("no_order"),
    "fixed_depth": _ablation_runner("fixed_depth"),
    "init_only": _ablation_runner("init_only"),
    "backtrack_only": _ablation_runner("backtrack_only"),
    "unify_target": _ablation_runner("unify_target"),
    "no_soft_facts_in_prompt": _ablation_runner("no_soft_facts_in_prompt"),
    "self_consistency": run_self_consistency,
}


def run_method(config_name: str, model: Optional[str] = None, method: str = "pllm",
               ablation: Optional[str] = None, run_index: int = 0, **kwargs) -> RunMetrics:
    """Single dispatch entry point used by the batch/timed runners.

    An ``ablation`` (a PLLM variant) takes precedence over ``method``; otherwise
    the named ``method`` runs. ``run_index`` is forwarded (random_edge seeds its
    RNG with it); extra kwargs (e.g. refine_budget, k) pass straight through.
    """
    call_kwargs = {"run_index": run_index, **kwargs}
    if ablation:
        if ablation not in ABLATIONS:
            raise KeyError(f"unknown ablation {ablation!r}; choices: {sorted(ABLATIONS)}")
        return ABLATIONS[ablation](config_name, model, **call_kwargs)
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; choices: {sorted(METHODS)}")
    return METHODS[method](config_name, model, **call_kwargs)