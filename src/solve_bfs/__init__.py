"""Legacy best-first meta-solving stack (superseded by ``solve``).

This is the ORIGINAL architecture: :class:`MetaSolver` repeatedly runs the
priority-queue :class:`solve.solver.Solver` and, between rounds, asks an
:class:`ExtensionStrategy` to widen the search — raise the depth limit, lower the
confidence floor, or add hypothesised clauses (:class:`LLMExtensionStrategy`).

It was replaced by the DFS stack in ``solve`` (``DFSMetaSolver`` +
``SearchGuidancePolicy``), which interleaves guidance with the search instead of
alternating solve/extend rounds. Nothing in the experiment pipeline instantiates
anything here any more: ``experiment.methods`` and ``experiment.tracking`` build
``DFSMetaSolver``, and the baselines verify with ``DFSSolver`` because the
priority-queue solver blows up combinatorially at the ceiling depth on the
recursive rules (see docs/BASELINES.md).

Kept, separated from the live code, as the reference for the earlier design.

NOTE ``solve.solver.Solver`` deliberately stayed in ``solve``: it is still live
code — ``experiment.groundtruth`` uses it on the complete KB to compute the
ground-truth proof, in a time-bounded subprocess.

Hypothesis parsing/insertion is NOT duplicated here: ``LLMExtensionStrategy``
calls ``solve.llm_search_guidance_policy.apply_clauses``, the same helper PLLM
and the baselines use.
"""

from solve_bfs.extension_strategy import ExtensionStrategy, TrivialExtensionStrategy
from solve_bfs.llm_extension_strategy import LLMExtensionStrategy, LLMPromptContext
from solve_bfs.metasolver import MetaSolver

__all__ = [
    "ExtensionStrategy",
    "LLMExtensionStrategy",
    "LLMPromptContext",
    "MetaSolver",
    "TrivialExtensionStrategy",
]
