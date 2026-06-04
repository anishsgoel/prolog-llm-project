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
from experiment.underground.prompts import BohemiaPromptBuilder, KrebsPromptBuilder, UndergroundPromptBuilder
from prolog.formula_parsing import parse_prolog_to_formula
from prolog.knowledge_base import KnowledgeBase, SoftKnowledgeBase
from prolog_llm.llm import LLMInterface
from solve import DFSMetaSolver, LLMSearchGuidancePolicy
from solve.solver import Solver

_CONFIGS_DIR = Path(__file__).parent / "configs"
ALL_CONFIGS: List[str] = sorted(p.stem for p in _CONFIGS_DIR.glob("*.yaml"))

_PROMPT_BUILDERS = {
    "underground": UndergroundPromptBuilder,
    "krebs": KrebsPromptBuilder,
    "bohemia": BohemiaPromptBuilder,
}


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

    # --- Ground truth comparison (Solver on the *full* KB) ---
    ground_truth_success: int = 0
    ground_truth_proof_depth: int = 0
    proof_depth_diff: int = 0      # proof_depth - ground_truth_proof_depth

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


# ---------------------------------------------------------------------------
# Main tracked-run entry point
# ---------------------------------------------------------------------------

def run_tracked(config_name: str) -> RunMetrics:
    """Run one experiment problem with full metrics collection.

    Equivalent to run.run() but returns a populated RunMetrics instead of
    printing results.  The original run.py is untouched and remains usable
    for quick debugging.
    """
    _cfg_mod.init_config()
    problem = ProblemConfig.from_yaml(_CONFIGS_DIR / f"{config_name}.yaml")

    kb_full = KnowledgeBase(problem.load_kb_text(_CONFIGS_DIR))
    kb_missing = kb_full.omit_facts(set(problem.omit_fact_ids))
    goal_formula = parse_prolog_to_formula(problem.goal)

    metrics = RunMetrics(
        config_name=config_name,
        goal=problem.goal,
        omit_fact_ids=str(problem.omit_fact_ids),
        model=_cfg_mod.MODEL,
        omitted_facts_total=len(problem.omit_fact_ids),
    )

    # --- Ground truth: classic BFS solver on the *complete* KB ---
    gt_result = Solver(kb_full, max_depth=30).solve(goal_formula)
    metrics.ground_truth_success = int(gt_result["success"])
    if gt_result["success"] and gt_result.get("proof") is not None:
        metrics.ground_truth_proof_depth = gt_result["proof"].depth

    # --- LLM-guided run with instrumented components ---
    tracked_llm = TrackedLLMInterface()
    prompt_builder = _PROMPT_BUILDERS[problem.prompt_builder](problem.solver)
    policy = TrackedLLMSearchGuidancePolicy(
        prompt_builder, tracked_llm=tracked_llm, solver_cfg=problem.solver
    )
    meta_solver = DFSMetaSolver(kb_missing, policy, solver_cfg=problem.solver)

    t0 = time.perf_counter()
    result = meta_solver.solve(goal_formula)
    metrics.wall_time_s = time.perf_counter() - t0

    # Outcome
    metrics.success = int(result["success"])
    metrics.proof_confidence = result.get("confidence", 0.0)
    if result["success"] and result.get("proof") is not None:
        metrics.proof_depth = result["proof"].depth
    metrics.max_depth_reached = result.get("max_depth", 0)
    metrics.min_confidence_final = result.get("min_confidence", 0.0)
    metrics.attempts_total = len(result.get("attempts", []))
    metrics.depth_start_estimate = policy.depth_estimate

    # LLM query counts
    metrics.llm_queries_total = tracked_llm.total_queries
    metrics.llm_queries_order = tracked_llm._counts.get("order", 0)
    metrics.llm_queries_backtrack = tracked_llm._counts.get("backtrack", 0)
    metrics.llm_queries_extend_on_init = tracked_llm._counts.get("extend_on_init", 0)
    metrics.llm_queries_estimate_depth = tracked_llm._counts.get("estimate_depth", 0)

    # Soft KB analysis
    final_soft_kb: Optional[SoftKnowledgeBase] = result.get("soft_kb")
    if final_soft_kb is not None:
        full_fact_atoms = {f.atom for f in kb_full.facts}
        omitted_atoms = {f.atom for f in kb_full.facts if f.num in set(problem.omit_fact_ids)}

        soft_facts = [f for f in final_soft_kb.facts if f.confidence < 1.0]
        soft_rules = [r for r in final_soft_kb.rules if r.confidence < 1.0]
        metrics.soft_facts_proposed = len(soft_facts)
        metrics.soft_rules_proposed = len(soft_rules)

        for sf in soft_facts:
            if sf.atom in full_fact_atoms:
                metrics.soft_facts_in_true_model += 1
            else:
                metrics.soft_facts_outside_true_model += 1
            if sf.atom in omitted_atoms:
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

    metrics.proof_depth_diff = metrics.proof_depth - metrics.ground_truth_proof_depth

    return metrics
