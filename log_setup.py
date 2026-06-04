"""Centralised logging configuration for the Prolog-LLM system.

Three named loggers are configured:

  ``pllm``         – general application logger (console output).
  ``pllm.llm``     – every LLM prompt and response, written to
                     ``logs/llm_prompts.log`` only (kept off the console).
  ``pllm.solver``  – verbose proof-search trace, written to
                     ``logs/solver_trace.log`` only.

Call :func:`setup_logging` once at start-up (``config.init_config`` does this).
Modules obtain their logger via :func:`get_logger`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_LOG_DIR = Path(__file__).resolve().parent / "logs"

APP_LOGGER = "pllm"
LLM_LOGGER = "pllm.llm"
SOLVER_LOGGER = "pllm.solver"

_configured = False


def setup_logging(
    verbose: bool = True,
    debug: bool = False,
    log_dir: Optional[Path] = None,
) -> None:
    """Configure the application, LLM, and solver loggers.

    Safe to call repeatedly: existing handlers are cleared first so that
    multiple ``init_config()`` calls (e.g. one per problem in a batch run)
    do not duplicate output.
    """
    global _configured

    directory = Path(log_dir) if log_dir else _LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)

    app = logging.getLogger(APP_LOGGER)
    llm = logging.getLogger(LLM_LOGGER)
    solver = logging.getLogger(SOLVER_LOGGER)

    app.setLevel(logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING)
    llm.setLevel(logging.DEBUG)
    solver.setLevel(logging.DEBUG if (verbose or debug) else logging.WARNING)

    # Child loggers write only to their own files, never to the console.
    for lg in (app, llm, solver):
        lg.propagate = False
        for handler in list(lg.handlers):
            lg.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    app.addHandler(console)

    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    llm_handler = logging.FileHandler(directory / "llm_prompts.log", encoding="utf-8")
    llm_handler.setFormatter(file_formatter)
    llm.addHandler(llm_handler)

    solver_handler = logging.FileHandler(directory / "solver_trace.log", encoding="utf-8")
    solver_handler.setFormatter(file_formatter)
    solver.addHandler(solver_handler)

    _configured = True


def get_logger(name: str = APP_LOGGER) -> logging.Logger:
    """Return a configured logger, initialising logging on first use."""
    if not _configured:
        setup_logging()
    return logging.getLogger(name)