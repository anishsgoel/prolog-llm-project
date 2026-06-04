import config
config.init_config()
from pathlib import Path
from cfg import ProblemConfig
from prolog.knowledge_base import KnowledgeBase
from prolog.formula_parsing import parse_prolog_to_formula
from solve.solver import Solver
from experiment.underground.prompts import BohemiaPromptBuilder

cfgdir = Path("src/experiment/configs")
problem = ProblemConfig.from_yaml(cfgdir / "bohemia_kings.yaml")
kb_full = KnowledgeBase(problem.load_kb_text(cfgdir))
kb_missing = kb_full.omit_facts(set(problem.omit_fact_ids))
goal = parse_prolog_to_formula(problem.goal)

print("RESULT facts:", len(kb_full.facts), "rules:", len(kb_full.rules), "omitted:", problem.omit_fact_ids)
full = Solver(kb_full, max_depth=30).solve(goal)
miss = Solver(kb_missing, max_depth=30).solve(goal)
print("RESULT FULL success:", full["success"], "depth:", full["proof"].depth if full["success"] else None)
print("RESULT MISSING success:", miss["success"])
pb = BohemiaPromptBuilder(problem.solver)
print("RESULT propose_facts:", pb.propose_facts, "builder:", problem.prompt_builder)
