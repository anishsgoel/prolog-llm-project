import sys, time
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import config
config.init_config()
config.VERBOSE = False
from pathlib import Path
from cfg import ProblemConfig
from prolog.knowledge_base import KnowledgeBase, SoftKnowledgeBase
from prolog.formula_parsing import parse_prolog_to_formula
from experiment.baseline_policies import candidate_atoms, add_candidate_facts, kb_constants
from experiment.methods import _prove
from itertools import combinations

CFG = Path("src/experiment/configs")
for name in ("london_directed", "london_t1_1edge_onpath_01"):
    p = ProblemConfig.from_yaml(CFG / f"{name}.yaml")
    kb_full = KnowledgeBase(p.load_kb_text(CFG))
    kb_missing = kb_full.omit_facts(set(p.omit_fact_ids))
    goal = parse_prolog_to_formula(p.goal)
    cands = candidate_atoms(kb_missing, p.propose_facts)
    print(f"=== {name} goal={p.goal} omit={p.omit_fact_ids} ceiling={p.solver.max_depth_ceiling}")
    print(f"  n_consts={len(kb_constants(kb_missing))} pool={len(cands)} k_max={min(len(p.omit_fact_ids)+1,3)}")
    print(f"  first 8 candidates: {cands[:8]}")
    omitted = [str(f.formula) for f in kb_full.facts if f.num in set(p.omit_fact_ids)]
    print(f"  omitted atoms: {omitted}")
    t0=time.perf_counter(); n=0; found=None
    for k in (1,):
        for d in combinations(cands, k):
            skb,_,_ = add_candidate_facts(SoftKnowledgeBase(kb_missing), d, 0.5)
            n+=1
            r = _prove(skb, goal, p.solver.max_depth_ceiling)
            if r.get("success"):
                found=(d, r.get("confidence"), r["proof"].depth, r["proof"].leaf_atoms()); break
        if found: break
    print(f"  probes={n} time={time.perf_counter()-t0:.2f}s -> delta={found[0] if found else None}")
    if found: print(f"  conf={found[1]} proof_depth={found[2]} leaves={found[3]}")
