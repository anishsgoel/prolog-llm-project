"""Background hypothesis generation."""

import json
import os
import re
import sys
from collections import deque
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prolog_llm.prolog_utils import (
    parse_predicate,
    is_variable,
    split_inline_comment,
    extract_first_json,
)
from prolog_llm.llm_interface import llm_json_only
import config


_BG_HYP_CACHE = {}


def _kb_signature_for_bg(kb: str) -> str:
    parts = []
    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\.\s*(.+)$", line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if not content:
            continue

        content = content.rstrip(".").strip()
        if content.startswith("connected(") or content.startswith("reachable("):
            parts.append(content)

    return str(hash("\n".join(parts)))


def _extract_connected_facts_and_stations(kb: str):
    hard_adj = {}
    hard_edges = set()
    stations = set()
    connected_facts = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\.\s*(.+)$", line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        content = (content or "").strip()
        if not content:
            continue

        if ":-" in content:
            continue

        atom0 = content.rstrip(".").strip()
        p = parse_predicate(atom0)
        if p and p[0] == "connected" and len(p[1]) == 2:
            a, b = p[1][0].strip(), p[1][1].strip()
            hard_adj.setdefault(a, []).append(b)
            hard_edges.add((a, b))
            stations.add(a)
            stations.add(b)
            connected_facts.append(f"connected({a}, {b}).")

    return hard_adj, hard_edges, stations, connected_facts


def _shortest_path_len_bounded(adj: dict, src: str, dst: str, max_depth: int = 25):
    if src == dst:
        return 0
    q = deque([(src, 0)])
    seen = {src}
    while q:
        node, d = q.popleft()
        if d >= max_depth:
            continue
        for nb in adj.get(node, []):
            if nb == dst:
                return d + 1
            if nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    return None


def _infer_preferred_atom_for_chain(goal: str, hard_adj: dict) -> Optional[str]:
    gp = parse_predicate(goal)
    if not gp or len(gp[1]) != 2:
        return None

    goal_src, goal_dst = gp[1][0].strip(), gp[1][1].strip()

    last = goal_src
    seen = set()
    while last in hard_adj and len(hard_adj[last]) == 1 and last not in seen:
        seen.add(last)
        last = hard_adj[last][0]

    if last and goal_dst:
        return f"connected({last}, {goal_dst})"
    return None


def _select_atoms_for_bg(unresolved_atoms, preferred_atom: Optional[str], max_atoms: int):
    atom_list = sorted(unresolved_atoms)

    ground = []
    for atom in atom_list:
        atom = atom.strip()
        if not atom or "(" not in atom or ")" not in atom:
            continue
        inside = atom.split("(", 1)[1].rsplit(")", 1)[0]
        if re.search(r"\b[A-Z_]\w*\b", inside):
            continue
        ground.append(atom)

    if preferred_atom:
        ground = [preferred_atom] + [a for a in ground if a != preferred_atom]

    if max_atoms is not None:
        ground = ground[:max_atoms]
    return ground


def _extract_constants_from_term(term: str) -> set:
    p = parse_predicate(term.strip().rstrip("."))
    if not p:
        return set()
    _, args = p
    return {a.strip() for a in args if a.strip() and not is_variable(a.strip())}


def _find_max_line_number_in_kb(kb: str) -> int:
    max_num = 0
    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r"^(\d+)\.\s*(.+)$", line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, _ = split_inline_comment(content_raw)
        if not content:
            continue

        num = int(m.group(1))
        if num > max_num:
            max_num = num
    return max_num


def _is_fact_clause(clause: str) -> bool:
    return ":-" not in clause


def _split_rule_clause(clause: str):
    clause = clause.strip()
    if clause.endswith("."):
        clause = clause[:-1]
    head_part, body_part = clause.split(":-", 1)
    return head_part.strip(), body_part.strip()


def attach_hypotheses_to_kb(kb: str, hypotheses):
    """Attach hypotheses to KB with penalties."""
    soft_facts = []
    soft_rules = []

    max_num = _find_max_line_number_in_kb(kb)
    next_num = max_num + 1

    for h in hypotheses:
        clause = (h.get("clause") or "").strip()
        if not clause:
            continue

        conf = float(h.get("confidence", 0.0))
        if not clause.endswith("."):
            clause = clause + "."

        penalty = 0.0
        if h.get("unknown_station"):
            penalty += config.SOFT_PENALTY_UNKNOWN_STATION

        if _is_fact_clause(clause):
            atom = clause.rstrip(".").strip()
            soft_facts.append((next_num, atom, conf, penalty))
        else:
            head, body_str = _split_rule_clause(clause)
            soft_rules.append((next_num, head, body_str, conf, penalty))

        next_num += 1

    return {"facts": soft_facts, "rules": soft_rules}


def generate_background_hypotheses_fast(
    goal: str,
    kb: str,
    hard_result: dict,
    predicate_comments: dict,
    max_atoms: Optional[int] = None,
    max_hyp_per_atom: Optional[int] = None,
    prompt_fact_limit: Optional[int] = None,
):
    """Generate background hypotheses using LLM."""

    if max_atoms is None:
        max_atoms = config.HYP_MAX_ATOMS
    if max_hyp_per_atom is None:
        max_hyp_per_atom = config.HYP_MAX_HYP_PER_ATOM
    if prompt_fact_limit is None:
        prompt_fact_limit = config.HYP_PROMPT_FACT_LIMIT

    unresolved_atoms = hard_result.get("unresolved_atoms", set()) if hard_result else set()
    if not unresolved_atoms:
        return []

    hard_adj, hard_edges, stations, connected_facts = _extract_connected_facts_and_stations(kb)

    stations_aug = set(stations)
    stations_aug |= _extract_constants_from_term(goal)
    for a in unresolved_atoms:
        stations_aug |= _extract_constants_from_term(a)

    preferred_atom = _infer_preferred_atom_for_chain(goal, hard_adj)
    atoms = _select_atoms_for_bg(unresolved_atoms, preferred_atom, max_atoms=max_atoms)

    if not atoms:
        print("[generate_background_hypotheses_fast] No suitable ground atoms.")
        return []

    kb_sig = _kb_signature_for_bg(kb)
    cache_key = (kb_sig, goal, tuple(atoms), max_hyp_per_atom, prompt_fact_limit)
    if cache_key in _BG_HYP_CACHE:
        return _BG_HYP_CACHE[cache_key]

    facts_for_prompt = connected_facts[:prompt_fact_limit]

    needed = {"connected/2"}
    for a in atoms:
        p = parse_predicate(a)
        if p:
            needed.add(f"{p[0]}/{len(p[1])}")

    semantic_lines = []
    for k in sorted(needed):
        if k in predicate_comments:
            semantic_lines.append(f"- {k}: {predicate_comments[k]}")
    semantic_hint_block = "\n".join(semantic_lines) if semantic_lines else "(none)"

    atoms_block = "\n".join([f"- {a}" for a in atoms])

    prompt = f"""
You are a cautious Prolog expert. You propose missing FACTS only.

GOAL:
  {goal}

HARD KB SNAPSHOT (partial; do NOT assume any other edges exist):
Connected facts (sample):
{chr(10).join(facts_for_prompt)}

Predicate semantics (ground truth):
{semantic_hint_block}

The proof failed because these subgoals could not be proven:
{atoms_block}

Task:
For each failed subgoal above, propose up to {max_hyp_per_atom} additional Prolog FACTS
that are likely true and would help prove the GOAL.

Constraints:
- Output ONLY connected/2 FACTS. No rules.
- Each fact MUST end with a period.
- Prefer using only station atoms that appear in the snapshot or the GOAL.
- If a suggested edge might be a "shortcut" (skips intermediate stops), lower confidence.

Return ONLY valid JSON in exactly this shape:

{{
  "by_atom": {{
    "connected(42nd_street, bryant_park)": [
      {{"clause":"connected(grand_central, bryant_park).","confidence":0.95}},
      {{"clause":"connected(42nd_street, bryant_park).","confidence":0.55}}
    ]
  }}
}}
""".strip()
    
    repair_schema = f"""{{
  "by_atom": {{
    "{atoms[0]}": [
      {{"clause":"connected(grand_central, bryant_park).","confidence":0.95}},
      {{"clause":"connected(42nd_street, bryant_park).","confidence":0.55}}
    ]
  }}
}}"""

    raw = llm_json_only(prompt, repair_schema).strip()
    if config.DEBUG:
        print("\n[DEBUG generate_background_hypotheses_fast] raw:\n", raw)

    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[generate_background_hypotheses_fast] JSON parse error:", e)
        print("Raw LLM output:", raw[:800])
        _BG_HYP_CACHE[cache_key] = []
        return []

    by_atom = data.get("by_atom", {})
    if not isinstance(by_atom, dict):
        _BG_HYP_CACHE[cache_key] = []
        return []

    def norm_clause(cl: str) -> Optional[str]:
        if not cl:
            return None
        cl = cl.strip()
        if not cl.endswith("."):
            cl += "."
        atom_str = cl.rstrip(".").strip()
        p = parse_predicate(atom_str)
        if not (p and p[0] == "connected" and len(p[1]) == 2):
            return None
        u, v = p[1][0].strip(), p[1][1].strip()
        if u == v:
            return None
        return f"connected({u}, {v})."

    def shortcut_meta(u: str, v: str):
        d = _shortest_path_len_bounded(hard_adj, u, v, max_depth=config.DEFAULT_MAX_DEPTH_SHORTCUT)
        is_shortcut = (d is not None and d >= 2)
        return is_shortcut, d

    out = []
    for atom, proposals in by_atom.items():
        if not isinstance(proposals, list):
            continue

        for item in proposals[:max_hyp_per_atom]:
            if not isinstance(item, dict):
                continue

            clause = norm_clause(item.get("clause", ""))
            if clause is None:
                continue

            try:
                conf = float(item.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            p = parse_predicate(clause.rstrip("."))
            u, v = p[1][0].strip(), p[1][1].strip()

            if (u, v) in hard_edges:
                continue

            is_shortcut, d = shortcut_meta(u, v)
            if is_shortcut:
                continue

            unknown_station = bool(stations_aug) and (u not in stations_aug or v not in stations_aug)

            out.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom,
                "is_shortcut": is_shortcut,
                "shortcut_len": d,
                "unknown_station": unknown_station,
            })

    dedup = {}
    for h in out:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    result = list(dedup.values())
    _BG_HYP_CACHE[cache_key] = result
    return result
