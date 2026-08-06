"""Time-bounded execution of a full tracked evaluation.

Each batch evaluation (one config x model run) makes blocking LLM calls and runs
an unbounded proof search, so a single problem can run far longer than wanted.
To cap it we run the whole evaluation in a separate process and terminate it
after a wall-clock budget; on timeout the caller records the run as a give-up.

A subprocess is used (not a thread or ``signal.alarm``) because the work is a
mix of CPU-bound search and blocking network calls that must be *killable* on
Windows, where ``signal.alarm`` is unavailable and threads cannot be force-
terminated.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, Optional

from experiment.groundtruth import _ensure_pythonpath
from log_setup import ERROR_LOGGER, get_logger

_err_log = get_logger(ERROR_LOGGER)


def _eval_worker(config_name: str, model: Optional[str], method: str,
                 ablation: Optional[str], run_index: int, queue: "mp.Queue") -> None:
    """Child process: run one tracked evaluation and return its metrics dict.

    ``method`` / ``ablation`` / ``run_index`` are passed as ARGUMENTS (never via
    module globals): the ``spawn`` child is a fresh interpreter that does not
    inherit the parent's globals, so the chosen baseline must travel through the
    process boundary explicitly.
    """
    try:
        from experiment.methods import run_method

        metrics = run_method(config_name, model=model, method=method,
                             ablation=ablation, run_index=run_index)
        queue.put(("ok", metrics.to_dict()))
    except Exception as exc:  # report instead of dying silently
        _err_log.exception("evaluation worker failed: config=%s model=%s method=%s ablation=%s",
                           config_name, model, method, ablation)
        queue.put(("error", f"{exc.__class__.__name__}: {exc}"))


def run_tracked_timed(
    config_name: str,
    model: Optional[str],
    timeout_s: float,
    method: str = "pllm",
    ablation: Optional[str] = None,
    run_index: int = 0,
) -> Dict[str, Any]:
    """Run one tracked evaluation with a wall-clock limit.

    Returns one of:
      ``{"status": "ok", "metrics": <RunMetrics.to_dict()>}``
      ``{"status": "error", "message": <str>}``
      ``{"status": "timeout"}``
    """
    _ensure_pythonpath()
    ctx = mp.get_context("spawn")
    queue: "mp.Queue" = ctx.Queue()
    proc = ctx.Process(target=_eval_worker,
                       args=(config_name, model, method, ablation, run_index, queue))
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"status": "timeout"}

    try:
        kind, payload = queue.get_nowait()
    except Exception:
        _err_log.exception("run_tracked_timed: worker finished without a result")
        return {"status": "error", "message": "evaluation worker died without a result"}

    if kind == "ok":
        return {"status": "ok", "metrics": payload}
    return {"status": "error", "message": payload}