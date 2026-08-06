"""DFS proof search and LLM-guided meta-solving (the live stack).

The earlier best-first architecture (``MetaSolver`` + ``ExtensionStrategy`` +
``LLMExtensionStrategy``) now lives in the ``solve_bfs`` package. ``Solver``
itself stays here: it is still used by ``experiment.groundtruth`` to prove the
goal on the complete KB.
"""

from solve.dfssolver import DFSSolver
from solve.dfs_metasolver import DFSMetaSolver
from solve.goalnode import GoalNode
from solve.prompt_buidler import PromptBuilder, LLMSearchGuidancePromptContext
from solve.llm_search_guidance_policy import LLMSearchGuidancePolicy
from solve.search_guidance_policy import SearchGuidancePolicy, TrivialSearchGuidancePolicy
from solve.solver import Solver

__all__ = [
    "DFSSolver",
    "DFSMetaSolver",
    "GoalNode",
    "PromptBuilder",
    "LLMSearchGuidancePromptContext",
    "LLMSearchGuidancePolicy",
    "SearchGuidancePolicy",
    "Solver",
    "TrivialSearchGuidancePolicy",
]
