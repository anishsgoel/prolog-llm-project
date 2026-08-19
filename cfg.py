"""Structured, type-safe configuration for the Prolog-LLM system.

Three config groups
-------------------
GlobalConfig  – LLM backend settings and system-wide limits.
SolverConfig  – DFS meta-solver and guidance-policy hyper-parameters.
ProblemConfig – Problem definition (goal, KB, omitted facts, domain).
               Supports multiple variants via ``model_copy(update={...})``.

YAML loading
------------
Each model exposes a ``from_yaml(path)`` class-method.  Problems can be
overridden in YAML:

    base = ProblemConfig.from_yaml("configs/london.yaml")
    harder = base.model_copy(update={"omit_fact_ids": [4, 6, 9]})
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# LLM / global settings
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    model: str = "gpt-oss:20b"
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    num_predict: int = Field(256, ge=1)
    stop_tokens: List[str] = Field(
        default_factory=lambda: ["\n\n", "\nWe ", "\nI ", "\nExplanation", "```"]
    )


class GlobalConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    debug: bool = False
    verbose: bool = True
    hyp_prompt_fact_limit: int = Field(25, ge=1)
    # Cap on HARD facts shown in a prompt, mirroring hyp_prompt_fact_limit for
    # soft ones. A KB at or under this size is passed through untouched, so
    # every pre-existing config keeps its exact old prompts; only the large
    # London tiers (up to 612 facts) are filtered. When it bites, facts are
    # chosen by goal relevance, not by truncating the list - see
    # solve.llm_search_guidance_policy.select_hard_facts.
    hard_prompt_fact_limit: int = Field(400, ge=1)
    # Radius of that goal-relevant subgraph, in hops over the constant
    # co-occurrence graph, measured from the constants in the goal and in the
    # currently open subgoals.
    hard_prompt_fact_hops: int = Field(3, ge=1)
    soft_bfs_max_solutions: int = Field(25, ge=1)
    path_aggregation: Literal["min", "max"] = "min"
    default_max_depth: int = Field(10, ge=1)
    default_max_depth_shortcut: int = Field(30, ge=1)
    hyp_max_atoms: int = Field(1, ge=1)
    hyp_max_hyp_per_atom: int = Field(1, ge=1)
    # --- system-wide solver behaviour ---
    # Proof selection objective. "max_confidence" (default) keeps the existing
    # behaviour; "min_hypotheses" prefers the shallowest all-hard-facts proof and
    # only falls back to soft search, picking the proof with the fewest
    # hypothesised leaves.
    proof_objective: Literal["max_confidence", "min_hypotheses"] = "max_confidence"
    # When True, the LLM judges each proposed clause (keep/drop) before it enters
    # the soft KB. Default keeps the current behaviour (no self-critique pass).
    verify_hypotheses: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GlobalConfig":
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))


# ---------------------------------------------------------------------------
# Solver / guidance-policy parameters
# ---------------------------------------------------------------------------

class SolverConfig(BaseModel):
    max_depth_ceiling: int = Field(30, ge=1)
    confidence_tolerance: float = Field(0.05, ge=0.0, le=1.0)
    max_binary_search_steps: int = Field(8, ge=1)
    max_hypotheses: int = Field(6, ge=1)
    allow_soft_rules: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SolverConfig":
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

class ProblemConfig(BaseModel):
    name: str
    goal: str
    kb_file: str
    # base relation(s) the LLM may hypothesise. Accepts a single value
    # ("connected/2") or a list (["targets/2", "implicated/2"]); both are
    # normalised to a list.
    propose_facts: List[str]
    # Whether the LLM may introduce constants not already present in the program.
    # This is a property of the problem (e.g. recovering a hidden node), so it
    # lives here rather than in the shared SolverConfig.
    allow_new_constants: bool = False
    omit_fact_ids: List[int] = Field(default_factory=list)
    solver: SolverConfig = Field(default_factory=SolverConfig)

    @field_validator("propose_facts", mode="before")
    @classmethod
    def _wrap_propose_facts(cls, value: Union[str, List[str]]) -> List[str]:
        return [value] if isinstance(value, str) else value

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProblemConfig":
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def load_kb_text(self, base_dir: str | Path) -> str:
        """Load the knowledge base text from kb_file relative to base_dir."""
        return Path(base_dir, self.kb_file).read_text()

    def variant(self, **overrides) -> "ProblemConfig":
        """Return a modified copy of this problem, e.g. with different omitted facts."""
        return self.model_copy(update=overrides)