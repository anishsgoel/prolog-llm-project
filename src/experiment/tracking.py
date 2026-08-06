"""Instrumented wrappers and metrics collection for experiment tracking.

Provides:
  - RunMetrics   – dataclass holding all statistics for one run
  - TrackedLLMInterface       – counts generate() calls by type
  - TrackedLLMSearchGuidancePolicy – sets call-type tag and records depth estimate
  - run_tracked(config_name)  – drop-in replacement for run.run(); returns RunMetrics
"""

from __future__ import annotations

import dataclasses
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- path setup (mirrors run.py) ----
_SRC_DIR = Path(__file__).parent.parent        # …/src
_PROJECT_ROOT = _SRC_DIR.parent                # …/prolog-llm-project
for _p in (str(_SRC_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as _cfg_mod
from cfg import ProblemConfig
from experiment.groundtruth import GROUND_TRUTH_TIMEOUT_S, solve_ground_truth
from experiment.prompts import PrologPromptBuilder
from prolog.formula_parsing import parse_prolog_to_formula
from prolog.knowledge_base import KnowledgeBase, SoftKnowledgeBase
from prolog_llm.llm import LLMInterface
from solve import DFSMetaSolver, LLMSearchGuidancePolicy

_CONFIGS_DIR = Path(__file__).parent / "configs"
ALL_CONFIGS: List[str] = sorted(p.stem for p in _CONFIGS_DIR.glob("*.yaml"))

# Value recorded for ground-truth-derived metrics when the ground-truth search
# exceeds its time budget. Flows through to CSV/xlsx as "NA"; in MLflow it is
# logged as a param (since it cannot be cast to a float metric).
NA = "NA"

# ---------------------------------------------------------------------------
# RunMetrics
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    """Statistics collected for a single experiment run.

    Numeric fields use int/float so they can be logged directly as MLflow
    metrics.  String fields (identity + list params) go into MLflow params.
    """

    # --- Identity ---
    config_name: str = ""
    goal: str = ""
    omit_fact_ids: str = ""        # serialised as "[4]" for CSV / MLflow params
    model: str = ""

    # --- Outcome ---
    success: int = 0               # 0 / 1
    proof_confidence: float = 0.0
    proof_depth: int = 0           # GoalNode.depth of the returned proof node
    proof: str = ""                # string form of the proof node returned by the meta-solver

    # --- Soundness of the returned proof (against the full model) ---
    proof_num_leaves: int = 0           # leaves in the returned proof
    proof_hallucinated_leaves: int = 0  # returned-proof leaves NOT in the full KB
    proof_is_sound: int = 0             # 1 iff success AND 0 hallucinated leaves
    valid_success: int = 0              # 1 iff success AND proof_is_sound

    # --- Ground truth comparison (Solver on the *full* KB) ---
    ground_truth_success: int = 0
    ground_truth_proof_depth: int = 0
    ground_truth_proof: str = ""   # string form of the proof found on the full KB
    proof_depth_diff: int = 0      # proof_depth - ground_truth_proof_depth
    proof_matches_gt: int = 0      # 1 iff leaf-atom set == ground-truth leaf-atom set; "NA" if GT timed out

    # --- Soft-KB analysis ---
    soft_facts_proposed: int = 0          # hypothesised facts in the final soft KB
    soft_rules_proposed: int = 0          # hypothesised rules in the final soft KB
    soft_facts_in_true_model: int = 0     # proposed facts that exist verbatim in the full KB
    soft_facts_outside_true_model: int = 0  # proposed facts *not* in the full KB (hallucinations)
    omitted_facts_recovered: int = 0      # proposed facts that match an omitted fact exactly
    omitted_facts_total: int = 0          # |omit_fact_ids|  (recall denominator)
    precision_hypothesized: float = 0.0   # soft_facts_in_true_model / soft_facts_proposed
    recall_omitted: float = 0.0           # omitted_facts_recovered / omitted_facts_total
    hallucination_rate: float = 0.0       # soft_facts_outside_true_model / soft_facts_proposed

    # --- Search statistics ---
    attempts_total: int = 0        # total DFSSolver.solve() calls (across depth & conf sweep)
    depth_start_estimate: int = 0  # LLM's estimate_depth() answer
    max_depth_reached: int = 0     # deepest depth limit tried
    min_confidence_final: float = 0.0   # best (highest) confidence threshold that yielded a proof

    # --- LLM query counts ---
    llm_queries_total: int = 0
    llm_queries_order: int = 0
    llm_queries_backtrack: int = 0
    llm_queries_extend_on_init: int = 0
    llm_queries_estimate_depth: int = 0
    llm_queries_verify: int = 0
    # Extra call types used by the LLM baselines (llm_oneshot / logic_lm). They
    # stay 0 for the PLLM run so its numbers are unchanged; adding them only
    # appends two all-zero columns for PLLM. Annotated as plain ``int`` so they
    # are picked up as numeric MLflow metrics (batch_run derives that from the
    # dataclass field type, which is a *string* here because of
    # ``from __future__ import annotations``).
    llm_queries_formalize: int = 0
    llm_queries_refine: int = 0

    # --- Timing ---
    wall_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Tracked LLM interface
# ---------------------------------------------------------------------------

class TrackedLLMInterface(LLMInterface):
    """LLMInterface that counts every generate() call, labelled by call type."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._call_type: str = "unknown"
        self._counts: Dict[str, int] = defaultdict(int)

    def set_call_type(self, t: str) -> None:
        self._call_type = t

    @property
    def total_queries(self) -> int:
        return sum(self._counts.values())

    def generate(self, *args, **kwargs) -> str:  # type: ignore[override]
        self._counts[self._call_type] += 1
        return super().generate(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tracked guidance policy
# ---------------------------------------------------------------------------

class TrackedLLMSearchGuidancePolicy(LLMSearchGuidancePolicy):
    """Wraps LLMSearchGuidancePolicy to tag call types and record depth estimate."""

    def __init__(
        self,
        llm_search_guidance,
        tracked_llm: TrackedLLMInterface,
        solver_cfg=None,
    ):
        super().__init__(llm_search_guidance, llm=tracked_llm, solver_cfg=solver_cfg)
        self._tracked_llm = tracked_llm
        self.depth_estimate: int = 0

    def order_goals(self, *args, **kwargs):
        self._tracked_llm.set_call_type("order")
        return super().order_goals(*args, **kwargs)

    def extend_on_backtrack(self, *args, **kwargs):
        self._tracked_llm.set_call_type("backtrack")
        return super().extend_on_backtrack(*args, **kwargs)

    def extend_on_init(self, *args, **kwargs):
        self._tracked_llm.set_call_type("extend_on_init")
        return super().extend_on_init(*args, **kwargs)

    def estimate_depth(self, *args, **kwargs) -> int:
        self._tracked_llm.set_call_type("estimate_depth")
        depth = super().estimate_depth(*args, **kwargs)
        self.depth_estimate = depth
        return depth

    def _verify_clauses(self, *args, **kwargs):
        self._tracked_llm.set_call_type("verify")
        return super()._verify_clauses(*args, **kwargs)


# ---------------------------------------------------------------------------
# Main tracked-run entry point
# ---------------------------------------------------------------------------

def build_problem(config_name: str, model: Optional[str] = None):
    """Load a problem config and prepare everything the solvers need.

    Extracted from ``run_tracked`` so every baseline in ``experiment.methods``
    starts from an identical setup (KB, ablated KB, goal, ground truth). Runs
    the classic solver on the *complete* KB under a wall-clock limit to obtain
    the ground truth.

    Returns ``(problem, kb_text, kb_full, kb_missing, goal_formula, gt)`` where
    ``gt`` is the tuple ``(success, depth, proof_str, leaf_atoms)`` from
    ``solve_ground_truth`` or ``None`` if the ground-truth search timed out.

    Side effects (deliberately mirroring the old ``run_tracked``): calls
    ``config.init_config()`` (which reloads global.yaml and resets
    ``PROOF_OBJECTIVE`` / ``VERIFY_HYPOTHESES``) and overrides ``config.MODEL``.
    Callers that flip config knobs for an ablation must do so *after* this
    returns (and, for ``VERIFY_HYPOTHESES``, before constructing the policy).
    """
    _cfg_mod.init_config()
    if model:
        _cfg_mod.MODEL = model
    problem = ProblemConfig.from_yaml(_CONFIGS_DIR / f"{config_name}.yaml")

    kb_text = problem.load_kb_text(_CONFIGS_DIR)
    kb_full = KnowledgeBase(kb_text)
    kb_missing = kb_full.omit_facts(set(problem.omit_fact_ids))
    goal_formula = parse_prolog_to_formula(problem.goal)

    # --- Ground truth: classic solver on the *complete* KB, time-bounded ---
    # If the search does not finish within the budget it is killed and the
    # ground-truth metrics are recorded as NA.
    gt = solve_ground_truth(kb_text, problem.goal, max_depth=30, timeout_s=GROUND_TRUTH_TIMEOUT_S)
    return problem, kb_text, kb_full, kb_missing, goal_formula, gt


def compute_metrics(
    config_name: str,
    model: str,
    problem: ProblemConfig,
    kb_full: KnowledgeBase,
    kb_missing: KnowledgeBase,
    result: Dict[str, Any],
    tracked_llm: Optional[TrackedLLMInterface],
    policy: Optional[Any],
    gt: Optional[tuple],
    wall_time_s: float,
) -> RunMetrics:
    """Compute the full, comparable RunMetrics for ONE run of ANY method.

    ``result`` is a plain dict shaped like ``DFSMetaSolver.solve`` output
    (keys ``success, proof, confidence, soft_kb, max_depth, min_confidence,
    attempts``); non-PLLM baselines synthesise it. ``tracked_llm`` and
    ``policy`` may be ``None`` (non-LLM / non-policy methods).

    This is the block that used to live inline in ``run_tracked``; the only
    change is that the soft-KB precision/recall/hallucination comparison now
    canonicalises atoms through ``AtomicFormula.__str__`` on BOTH sides
    (matching the soundness block just above it), instead of comparing the raw
    ``Fact.atom`` text. Every KB in this repo writes facts with a single
    ``", "`` between arguments, so the two are numerically identical for the
    PLLM run; canonicalising just makes precision/recall robust to spacing so
    the fact-proposing baselines (which emit canonical atoms) are scored on the
    same footing rather than reading 0.
    """
    metrics = RunMetrics(
        config_name=config_name,
        goal=problem.goal,
        omit_fact_ids=str(problem.omit_fact_ids),
        model=model,
        omitted_facts_total=len(problem.omit_fact_ids),
    )

    ground_truth_timed_out = gt is None
    gt_success = False
    gt_leaves: List[str] = []
    if ground_truth_timed_out:
        metrics.ground_truth_success = NA
        metrics.ground_truth_proof_depth = NA
        metrics.ground_truth_proof = NA
    else:
        gt_success, gt_depth, gt_proof, gt_leaves = gt
        metrics.ground_truth_success = int(gt_success)
        if gt_success:
            metrics.ground_truth_proof_depth = gt_depth
            metrics.ground_truth_proof = gt_proof

    # Outcome
    metrics.success = int(result["success"])
    metrics.proof_confidence = result.get("confidence", 0.0)
    if result.get("proof") is not None:
        metrics.proof = str(result["proof"])
        if result["success"]:
            metrics.proof_depth = result["proof"].depth
    metrics.max_depth_reached = result.get("max_depth", 0)
    metrics.min_confidence_final = result.get("min_confidence", 0.0)
    metrics.attempts_total = len(result.get("attempts", []))
    metrics.depth_start_estimate = getattr(policy, "depth_estimate", 0) if policy is not None else 0

    # Soundness: is every leaf of the returned proof a real fact in the full model?
    if result["success"] and result.get("proof") is not None:
        leaves = result["proof"].leaf_atoms()
        metrics.proof_num_leaves = len(leaves)
        # Normalise both sides through AtomicFormula.__str__ so spacing differences
        # in the raw KB text do not cause spurious "hallucinated" mismatches.
        full_fact_strs = {str(f.formula) for f in kb_full.facts}
        metrics.proof_hallucinated_leaves = sum(1 for s in leaves if s not in full_fact_strs)
        metrics.proof_is_sound = int(metrics.proof_hallucinated_leaves == 0)
        metrics.valid_success = int(bool(result["success"]) and bool(metrics.proof_is_sound))

    # LLM query counts (guarded: non-LLM baselines pass tracked_llm=None)
    if tracked_llm is not None:
        counts = tracked_llm._counts
        metrics.llm_queries_total = tracked_llm.total_queries
        metrics.llm_queries_order = counts.get("order", 0)
        metrics.llm_queries_backtrack = counts.get("backtrack", 0)
        metrics.llm_queries_extend_on_init = counts.get("extend_on_init", 0)
        metrics.llm_queries_estimate_depth = counts.get("estimate_depth", 0)
        metrics.llm_queries_verify = counts.get("verify", 0)
        metrics.llm_queries_formalize = counts.get("formalize", 0)
        metrics.llm_queries_refine = counts.get("refine", 0)

    # Soft KB analysis
    final_soft_kb: Optional[SoftKnowledgeBase] = result.get("soft_kb")
    if final_soft_kb is not None:
        omit_ids = set(problem.omit_fact_ids)
        full_fact_strs = {str(f.formula) for f in kb_full.facts}
        omitted_strs = {str(f.formula) for f in kb_full.facts if f.num in omit_ids}

        soft_facts = [f for f in final_soft_kb.facts if f.confidence < 1.0]
        soft_rules = [r for r in final_soft_kb.rules if r.confidence < 1.0]
        metrics.soft_facts_proposed = len(soft_facts)
        metrics.soft_rules_proposed = len(soft_rules)

        for sf in soft_facts:
            sf_str = str(sf.formula)
            if sf_str in full_fact_strs:
                metrics.soft_facts_in_true_model += 1
            else:
                metrics.soft_facts_outside_true_model += 1
            if sf_str in omitted_strs:
                metrics.omitted_facts_recovered += 1

        if metrics.soft_facts_proposed > 0:
            metrics.precision_hypothesized = (
                metrics.soft_facts_in_true_model / metrics.soft_facts_proposed
            )
            metrics.hallucination_rate = (
                metrics.soft_facts_outside_true_model / metrics.soft_facts_proposed
            )
        if metrics.omitted_facts_total > 0:
            metrics.recall_omitted = (
                metrics.omitted_facts_recovered / metrics.omitted_facts_total
            )

    if ground_truth_timed_out:
        metrics.proof_depth_diff = NA
        metrics.proof_matches_gt = NA
    else:
        metrics.proof_depth_diff = metrics.proof_depth - metrics.ground_truth_proof_depth
        if gt_success and result["success"] and result.get("proof") is not None:
            metrics.proof_matches_gt = int(set(result["proof"].leaf_atoms()) == set(gt_leaves))
        else:
            metrics.proof_matches_gt = 0

    metrics.wall_time_s = wall_time_s
    return metrics


def run_tracked(config_name: str, model: Optional[str] = None) -> RunMetrics:
    """Run one experiment problem (PLLM) with full metrics collection.

    Equivalent to run.run() but returns a populated RunMetrics instead of
    printing results.  The original run.py is untouched and remains usable
    for quick debugging.

    If ``model`` is given it overrides the configured LLM model for this run,
    so the same problem can be evaluated across several models.
    """
    problem, kb_text, kb_full, kb_missing, goal_formula, gt = build_problem(config_name, model)
    if gt is None:
        print(f"  ground-truth search exceeded {GROUND_TRUTH_TIMEOUT_S}s -> recording NA")

    # --- LLM-guided run with instrumented components ---
    tracked_llm = TrackedLLMInterface(model=_cfg_mod.MODEL)
    prompt_builder = PrologPromptBuilder(problem.propose_facts, problem.solver,
                                         allow_new_constants=problem.allow_new_constants)
    policy = TrackedLLMSearchGuidancePolicy(
        prompt_builder, tracked_llm=tracked_llm, solver_cfg=problem.solver
    )
    meta_solver = DFSMetaSolver(kb_missing, policy, solver_cfg=problem.solver)

    t0 = time.perf_counter()
    result = meta_solver.solve(goal_formula)
    wall_time_s = time.perf_counter() - t0

    return compute_metrics(
        config_name, _cfg_mod.MODEL, problem, kb_full, kb_missing,
        result, tracked_llm, policy, gt, wall_time_s,
    )
