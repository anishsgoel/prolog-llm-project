#!/usr/bin/env python
"""Universal experiment entry point.

Usage:
    python src/experiment/run.py <config_name>

where <config_name> is the YAML stem under src/experiment/configs/,
e.g. london_directed, krebs_cycle, underground_directed2.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from cfg import ProblemConfig
from experiment.underground.prompts import PrologPromptBuilder
from prolog.formula_parsing import parse_prolog_to_formula
from prolog.knowledge_base import KnowledgeBase
from solve import DFSMetaSolver, LLMSearchGuidancePolicy
from solve.solver import Solver

_CONFIGS_DIR = Path(__file__).parent / "configs"


def run(config_name: str) -> None:
    config.init_config()
    problem = ProblemConfig.from_yaml(_CONFIGS_DIR / f"{config_name}.yaml")

    kb_obj = KnowledgeBase(problem.load_kb_text(_CONFIGS_DIR))
    kb_missing_obj = kb_obj.omit_facts(set(problem.omit_fact_ids))
    goal_formula = parse_prolog_to_formula(problem.goal)

    print(f"Problem: {problem.name}  goal: {problem.goal}  omitted: {problem.omit_fact_ids}")
    print(f"Solver config: {problem.solver.model_dump()}\n")

    s = Solver(kb_missing_obj, 10)
    result = s.solve(goal_formula)
    print(f"Baseline solver: success={result['success']}\n")

    prompt_builder = PrologPromptBuilder(problem.propose_facts, problem.solver)
    s = DFSMetaSolver(kb_missing_obj, LLMSearchGuidancePolicy(prompt_builder, solver_cfg=problem.solver), solver_cfg=problem.solver)
    result = s.solve(goal_formula)
    print(f"DFS meta-solver: success={result['success']}, confidence={result['confidence']}")
    print(f"Proof: {result['proof']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Prolog-LLM experiment.")
    parser.add_argument("config", help="Config name (YAML stem), e.g. london_directed")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()