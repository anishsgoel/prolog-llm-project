"""Time-bounded ground-truth solving.

The ground-truth proof is computed with the plain ``Solver`` on the *complete*
KB.  On some knowledge bases (e.g. symmetric/undirected interaction graphs) that
search can take a very long time.  To keep batch runs moving, we run it in a
separate process and terminate it after a timeout; on timeout the caller records
NA for the ground-truth metrics.

A subprocess is used (rather than a thread or ``signal.alarm``) because the
search is CPU-bound pure Python and must be *killable* on Windows, where
``signal.alarm`` is unavailable and threads cannot be force-terminated.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent   # …/src
_PROJECT_ROOT = _SRC_DIR.parent                     # …/prolog-llm-project

# Default wall-clock budget for the ground-truth search (seconds).
GROUND_TRUTH_TIMEOUT_S = 300  # 5 minutes


def _ensure_pythonpath() -> None:
    """Propagate the project import paths to spawned children via PYTHONPATH.

    ``spawn`` (the default on Windows) starts a fresh interpreter that does not
    inherit the parent's runtime ``sys.path`` edits, but it does inherit the
    environment, so the child can import ``config``/``prolog``/``solve``.
    """
    existing = os.environ.get("PYTHONPATH", "")
    existing_parts = existing.split(os.pathsep) if existing else []
    parts = [p for p in (str(_SRC_DIR), str(_PROJECT_ROOT)) if p not in existing_parts]
    os.environ["PYTHONPATH"] = os.pathsep.join(parts + existing_parts)


def _gt_worker(kb_text: str, goal_str: str, max_depth: int, queue: mp.Queue) -> None:
    """Child process: reconstruct the KB, solve, and return (success, depth)."""
    from prolog.knowledge_base import KnowledgeBase
    from prolog.formula_parsing import parse_prolog_to_formula
    from solve.solver import Solver

    kb = KnowledgeBase(kb_text)
    goal = parse_prolog_to_formula(goal_str)
    result = Solver(kb, max_depth=max_depth).solve(goal)
    success = bool(result.get("success"))
    proof = result.get("proof")
    depth = proof.depth if (success and proof is not None) else 0
    queue.put((success, int(depth)))


def solve_ground_truth(
    kb_text: str,
    goal_str: str,
    max_depth: int = 30,
    timeout_s: float = GROUND_TRUTH_TIMEOUT_S,
) -> Optional[Tuple[bool, int]]:
    """Solve the ground-truth goal with a wall-clock limit.

    Returns ``(success, proof_depth)`` if the search finishes in time, or
    ``None`` if it exceeds ``timeout_s`` (or the worker dies without a result),
    signalling the caller to record NA.
    """
    _ensure_pythonpath()
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_gt_worker, args=(kb_text, goal_str, max_depth, queue))
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None

    try:
        return queue.get_nowait()
    except Exception:
        return None