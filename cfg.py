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
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


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
    soft_bfs_max_solutions: int = Field(25, ge=1)
    path_aggregation: Literal["min", "max"] = "min"
    default_max_depth: int = Field(10, ge=1)
    default_max_depth_shortcut: int = Field(30, ge=1)
    hyp_max_atoms: int = Field(1, ge=1)
    hyp_max_hyp_per_atom: int = Field(1, ge=1)

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
    allow_soft_rules: bool = True
    allow_new_constants: bool = False

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
    prompt_builder: Literal["underground", "krebs"] = "underground"
    omit_fact_ids: List[int] = Field(default_factory=list)
    solver: SolverConfig = Field(default_factory=SolverConfig)

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