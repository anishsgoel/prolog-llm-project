import ollama
import heapq
from itertools import count
from collections import deque
import re
import json
from typing import Optional
import math

import config


# ============================================================
# Config / LLM setup
# ============================================================

config.init_config()

client = ollama.Client()
model = config.MODEL

DEBUG = config.DEBUG
VERBOSE = config.VERBOSE


def ask_llm(prompt: str) -> str:
    """
    Drop-in patch: prevents 'reasoning' rambles and forces a JSON payload.
    - Adds stop tokens to cut off commentary early.
    - Falls back to `thinking` field if `response` is empty (common with gpt-oss).
    - If client supports format='json', it will be used; otherwise it will ignore it.
    """
    try:
        kwargs = dict(
            model=model,
            prompt=prompt,
            options={
                "temperature": config.LLM_TEMPERATURE,
                "num_predict": config.LLM_NUM_PREDICT,
                "stop": config.LLM_STOP_TOKENS,
            },
        )

        # Best-effort JSON mode (supported by many Ollama builds / clients).
        # If your client doesn't accept it, we catch TypeError and retry without it.
        kwargs["format"] = "json"

        resp = client.generate(**kwargs)

    except TypeError:
        # Older client: no "format" kwarg
        try:
            kwargs.pop("format", None)
            resp = client.generate(**kwargs)
        except Exception as e:
            print("[ask_llm] Ollama generate() exception:", repr(e))
            return ""
    except Exception as e:
        print("[ask_llm] Ollama generate() exception:", repr(e))
        return ""

    # Some models put output into `thinking` and leave `response` empty
    answer = (resp.get("response") or "").strip()
    if not answer:
        answer = (resp.get("thinking") or "").strip()

    if not answer:
        print("[ask_llm] EMPTY response+thinking. Raw resp:", resp)
        return ""

    if "...done thinking." in answer:
        answer = answer.split("...done thinking.")[-1].strip()

    return answer



# ============================================================
# Helpers for parsing / JSON
# ============================================================

def extract_first_json(text: str) -> str:
    """
    Extract the first balanced {...} JSON object from possibly messy text.
    Brace-count scanning (non-greedy, stall-safe).
    """
    if not text:
        raise ValueError("No JSON object found (empty text).")

    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in: {text!r}")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    raise ValueError(f"Unclosed JSON object in: {text[start:start+200]!r}...")

def llm_json_only(prompt: str, repair_schema: str) -> str:
    """
    Calls ask_llm. If no JSON object is present, retries once with a strict JSON-only repair prompt.
    Returns raw text (which should contain JSON).
    """
    raw = ask_llm(prompt).strip()
    try:
        _ = extract_first_json(raw)
        return raw
    except Exception:
        pass

    # Retry: extremely strict "JSON only" instruction
    repair_prompt = f"""
Return ONLY valid JSON. No explanation, no prose, no markdown.

You MUST output JSON that matches this schema exactly:
{repair_schema}

If you cannot comply, output:
{{"by_atom":{{}}}}
""".strip()

    raw2 = ask_llm(repair_prompt).strip()
    return raw2

    
def split_inline_comment(s: str):
    """
    Split a string into (code, comment) at the first '#'.
    Returns comment WITHOUT the '#'. If no comment, comment=None.
    """
    if "#" not in s:
        return s.strip(), None
    code, comment = s.split("#", 1)
    code = code.strip()
    comment = comment.strip()
    return code, (comment if comment else None)


def parse_predicate(term: str):
    """
    Parse a simple Prolog predicate of the form:
        functor(arg1, arg2, ...)
    Returns (functor: str, args: list[str]) or None if parsing fails.
    """
    term = term.strip().rstrip(".")
    m = re.match(r"^([a-z_][a-zA-Z0-9_]*)\((.*)\)$", term)
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        args = [a.strip() for a in args_raw.split(",")]
    return functor, args


def is_variable(s: str) -> bool:
    """Prolog-ish variable check: starts with uppercase letter or '_'."""
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == "_")


def strip_inline_comment(s: str) -> str:
    return s.split("#", 1)[0].rstrip()


def is_ground_atom(atom: str) -> bool:
    """
    Groundness filter for metric collection: no variables in arg list.
    Keeps the solver logic unchanged; only affects what we log/hand to LLM.
    """
    p = parse_predicate(atom.strip().rstrip("."))
    if not p:
        return False
    _, args = p
    return all(not is_variable(a) for a in args)


# ============================================================
# Core Prolog helpers
# ============================================================

def check_exact_match(goal: str, fact: str) -> bool:
    return goal.strip().rstrip(".") == fact.strip().rstrip(".")


def unify_args(args_goal, args_fact, env=None):
    if env is None:
        env = {}

    if len(args_goal) != len(args_fact):
        return None

    for g, f in zip(args_goal, args_fact):
        g = g.strip()
        f = f.strip()

        g_is_var = is_variable(g)
        f_is_var = is_variable(f)

        if not g_is_var and not f_is_var:
            if g != f:
                return None
            continue

        if g_is_var and not f_is_var:
            if g in env:
                if env[g] != f:
                    return None
            else:
                env[g] = f
            continue

        if not g_is_var and f_is_var:
            if f in env:
                if env[f] != g:
                    return None
            else:
                env[f] = g
            continue

        if g_is_var and f_is_var:
            if g in env and f in env:
                if env[g] != env[f]:
                    return None
            elif g in env:
                env[f] = env[g]
            elif f in env:
                env[g] = env[f]
            continue

    return env


def unify_arg_lists(args_rule_head, args_goal):
    return unify_args(args_rule_head, args_goal, env={})


def unify_with_fact(goal: str, fact: str):
    parsed_goal = parse_predicate(goal)
    parsed_fact = parse_predicate(fact)
    if parsed_goal is None or parsed_fact is None:
        return None

    fun_g, args_g = parsed_goal
    fun_f, args_f = parsed_fact

    if fun_g != fun_f or len(args_g) != len(args_f):
        return None

    if check_exact_match(goal, fact):
        return {}

    env = unify_args(args_g, args_f, env={})
    if env is None:
        return None

    return env


def apply_bindings(goals, bindings):
    if not bindings or not goals:
        return goals

    new_goals = []
    for g in goals:
        parsed = parse_predicate(g)
        if parsed is None:
            new_goals.append(g)
            continue

        functor, args = parsed
        new_args = []
        for a in args:
            a_stripped = a.strip()
            if is_variable(a_stripped) and a_stripped in bindings:
                new_args.append(bindings[a_stripped])
            else:
                new_args.append(a_stripped)

        new_goals.append(f"{functor}({', '.join(new_args)})")

    return new_goals


def find_matching_rules_only(goal, rules_list):
    parsed_goal = parse_predicate(goal)
    if parsed_goal is None:
        return []
    fun_g, args_g = parsed_goal
    arity_g = len(args_g)

    matching = []
    for num, head, body in rules_list:
        parsed_head = parse_predicate(head)
        if parsed_head is None:
            continue
        fun_h, args_h = parsed_head
        if fun_h == fun_g and len(args_h) == arity_g:
            matching.append(num)
    return matching


def substitute_in_atom(atom: str, bindings: dict) -> str:
    parsed = parse_predicate(atom)
    if parsed is None:
        return atom

    functor, args = parsed
    new_args = []
    for a in args:
        a_stripped = a.strip()
        if is_variable(a_stripped) and a_stripped in bindings:
            new_args.append(bindings[a_stripped])
        else:
            new_args.append(a_stripped)

    return f"{functor}({', '.join(new_args)})"


def split_body_atoms(body_str: str):
    body_str = body_str.strip()
    atoms = []
    current = []
    depth = 0

    for ch in body_str:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == "," and depth == 0:
            atom = "".join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(ch)

    atom = "".join(current).strip()
    if atom:
        atoms.append(atom)

    return atoms


def get_subgoals(goal: str, rule_head: str, rule_body: str):
    parsed_goal = parse_predicate(goal)
    parsed_head = parse_predicate(rule_head)

    if parsed_goal is None or parsed_head is None:
        return None

    fun_g, args_g = parsed_goal
    fun_h, args_h = parsed_head

    if fun_g != fun_h or len(args_g) != len(args_h):
        return None

    bindings = unify_arg_lists(args_h, args_g)
    if bindings is None:
        return None

    body_str = rule_body.strip()
    if not body_str:
        return []

    body_atoms = split_body_atoms(body_str)
    if not body_atoms:
        return []

    subgoals = [substitute_in_atom(atom, bindings) for atom in body_atoms]
    return subgoals if subgoals else None


# ============================================================
# KB comment extraction (inline + full-line)
# ============================================================

def parse_kb_predicate_comments(kb: str):
    predicate_comments = {}
    pending_comments = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("#"):
            pending_comments.append(line.lstrip("#").strip())
            continue

        m = re.match(r"^(\d+)\.\s*(.+)$", line)
        if not m:
            continue

        content_raw = m.group(2).strip()
        content, inline_comment = split_inline_comment(content_raw)

        clause = content.strip().rstrip(".")
        head_str = clause.split(":-", 1)[0].strip()
        parsed = parse_predicate(head_str)
        if parsed is None:
            pending_comments = []
            continue

        functor, args = parsed
        key = f"{functor}/{len(args)}"

        combined = []
        if pending_comments:
            combined.append(" ".join(pending_comments))
        if inline_comment:
            combined.append(inline_comment)

        if combined:
            predicate_comments[key] = " ".join(combined).strip()

        pending_comments = []

    return predicate_comments


# ============================================================
# BFS Prolog engine (hard) + minimal metric collection
# ============================================================

def bfs_prolog_collect(goal: str, kb: str, max_depth: int = None):
    """
    Same solver logic as before, but metric collection is now defensible:
      - returns a single "blocking_atom" (ground) if possible
      - avoids logging/returning all dead-end Rule-8 branches
    """
    if max_depth is None:
        max_depth = config.DEFAULT_MAX_DEPTH
    facts = []
    rules = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        line = strip_inline_comment(line).strip()
        if not line:
            continue

        match = re.match(r"^(\d+)\.\s*(.+)$", line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)

            if not content:
                continue

            if ":-" in content:
                head, body = content.split(":-", 1)
                rules.append((num, head.strip(), body.strip().rstrip(".")))
            else:
                facts.append((num, content.rstrip(".")))

    queue = deque([(goal, [], [], 0)])
    visited = set()

    # -------- metrics (minimal + paper-friendly) --------
    expansions = 0
    best_blocking_atom = None
    best_blocking_depth = -1
    blocking_reason = None  # "no_progress" or "depth_cap"
    # ---------------------------------------------------

    if VERBOSE:
        print(f"\n[COLLECT] Goal: {goal}")
        print("-" * 40)

    while queue:
        current, remaining, path, depth = queue.popleft()

        expansions += 1

        if depth >= max_depth:
            # metric-only: consider this a possible blocker if ground
            if is_ground_atom(current) and depth > best_blocking_depth:
                best_blocking_atom = current
                best_blocking_depth = depth
                blocking_reason = "depth_cap"
            continue

        state = (current, tuple(remaining))
        if state in visited:
            continue
        visited.add(state)

        if VERBOSE:
            print(f"Depth {depth}: {current}")
            if remaining:
                print(f"  Remaining: {remaining}")

        progress = False

        # 1) Exact fact match
        for num, fact in facts:
            if check_exact_match(current, fact):
                if VERBOSE:
                    print(f"  ✓ Fact {num} matches exactly!")

                new_path = path + [f"Fact {num}"]
                if not remaining:
                    if VERBOSE:
                        print(f"✓✓ SUCCESS at depth {depth + 1}")
                    return {
                        "success": True,
                        "proof_path": new_path,
                        "unresolved_atoms": set(),
                        "metrics": {
                            "hard_expansions": expansions,
                            "blocking_atom": None,
                            "blocking_depth": None,
                            "blocking_reason": None,
                        }
                    }

                queue.append((remaining[0], remaining[1:], new_path, depth + 1))
                progress = True
                break

        if progress:
            continue

        # 2) Fact unification
        for num, fact in facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            progress = True

            if VERBOSE:
                print(f"  ✓ Fact {num}: {fact}")
                print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)
            new_path = path + [f"Fact {num}"]

            if not instantiated:
                if VERBOSE:
                    print(f"✓✓ SUCCESS at depth {depth + 1}")
                return {
                    "success": True,
                    "proof_path": new_path,
                    "unresolved_atoms": set(),
                    "metrics": {
                        "hard_expansions": expansions,
                        "blocking_atom": None,
                        "blocking_depth": None,
                        "blocking_reason": None,
                    }
                }

            queue.append((instantiated[0], instantiated[1:], new_path, depth + 1))

        if progress:
            continue

        # 3) Rules
        matching_rules = find_matching_rules_only(current, rules)
        if VERBOSE and matching_rules:
            print(f"  Matching rules: {matching_rules}")

        for rule_num in matching_rules:
            for num, head, body in rules:
                if num != rule_num:
                    continue
                subgoals = get_subgoals(current, head, body)
                if subgoals:
                    if VERBOSE:
                        print(f"  Rule {num}: → {subgoals}")
                    progress = True
                    all_goals = subgoals + remaining
                    queue.append((all_goals[0], all_goals[1:], path + [f"Rule {num}"], depth + 1))
                break

        if not progress:
            if VERBOSE:
                print(f"  ✗ No facts or rules apply to: {current}")

            # metric-only: treat only the deepest ground "no progress" atom as the blocker
            if is_ground_atom(current) and depth > best_blocking_depth:
                best_blocking_atom = current
                best_blocking_depth = depth
                blocking_reason = "no_progress"

    if VERBOSE:
        print("✗ FAILED (collect mode)")

    unresolved = set()
    if best_blocking_atom:
        unresolved.add(best_blocking_atom)

    return {
        "success": False,
        "proof_path": [],
        "unresolved_atoms": unresolved,
        "metrics": {
            "hard_expansions": expansions,
            "blocking_atom": best_blocking_atom,
            "blocking_depth": best_blocking_depth if best_blocking_atom else None,
            "blocking_reason": blocking_reason,
        }
    }


# ============================================================
# FAST background hypothesis generation (single-call + cache)
#   Bug patches:
#   (1) unknown_station now includes constants from GOAL + queried atoms
#       so goal-dst isn’t mislabeled “unknown” when removed from KB.
#   (2) prompt/JSON stays small: 1 atom, 2 hyps, 25 fact snapshot
#   (3) shortcut rejection remains hard (adjacent-only semantics)
# ============================================================

_BG_HYP_CACHE = {}  # (kb_sig, goal, tuple(atoms), max_hyp_per_atom, prompt_fact_limit) -> list[hypothesis]


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
    # deterministic ordering helps caching
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


def generate_background_hypotheses_fast(
    goal: str,
    kb: str,
    hard_result: dict,
    predicate_comments: dict,
    max_atoms: int = 1,
    max_hyp_per_atom: int = 2,
    prompt_fact_limit: int = 25,
):
    unresolved_atoms = hard_result.get("unresolved_atoms", set()) if hard_result else set()
    if not unresolved_atoms:
        return []

    hard_adj, hard_edges, stations, connected_facts = _extract_connected_facts_and_stations(kb)

    # BUG PATCH: treat goal constants (and the blocking atom constants) as “known”
    # so we don't penalize the model for using the target node that’s missing from KB facts.
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

    # Keep prompt close to original logic, but minimize size and ambiguity.
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
    if DEBUG:
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

            # skip duplicates already in hard KB
            if (u, v) in hard_edges:
                continue

            is_shortcut, d = shortcut_meta(u, v)
            if is_shortcut:
                # Hard reject shortcuts to match adjacent-only semantics
                continue

            # BUG PATCH: unknown station against augmented symbol table
            unknown_station = bool(stations_aug) and (u not in stations_aug or v not in stations_aug)

            out.append({
                "clause": clause,
                "confidence": conf,
                "from_atom": atom,
                "is_shortcut": is_shortcut,
                "shortcut_len": d,
                "unknown_station": unknown_station,
            })

    # Dedup keep best confidence
    dedup = {}
    for h in out:
        key = h["clause"]
        if key not in dedup or h["confidence"] > dedup[key]["confidence"]:
            dedup[key] = h

    result = list(dedup.values())
    _BG_HYP_CACHE[cache_key] = result
    return result


# ============================================================
# Hypothesis attachment
# ============================================================

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
    """
    Soft facts include penalty:
      (num, atom, conf, penalty)

    penalty used to deprioritize unknown stations.
    Shortcuts are already rejected earlier.
    """
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


# ============================================================
# SOFT BFS with confidence-first priority + penalty
# (unchanged solver logic; optional verbosity)
# ============================================================

def bfs_prolog_metro_soft(
    goal: str,
    kb: str,
    soft_kb,
    max_depth: Optional[int] = None,
    max_soft: Optional[int] = None,
    max_solutions: Optional[int] = None,
):
    if max_depth is None:
        max_depth = config.DEFAULT_MAX_DEPTH
    if max_solutions is None:
        max_solutions = config.SOFT_BFS_MAX_SOLUTIONS
    # Parse hard KB
    hard_facts = []
    hard_rules = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        line = strip_inline_comment(line).strip()
        if not line:
            continue

        match = re.match(r"^(\d+)\.\s*(.+)$", line)
        if match:
            num = int(match.group(1))
            content_raw = match.group(2).strip()
            content, _ = split_inline_comment(content_raw)
            content = (content or "").strip()
            if not content:
                continue

            if ":-" in content:
                head, body = content.split(":-", 1)
                hard_rules.append((num, head.strip(), body.strip().rstrip(".")))
            else:
                hard_facts.append((num, content.rstrip(".")))

    # Soft KB unpack
    soft_facts = soft_kb.get("facts", [])
    soft_rules = soft_kb.get("rules", [])
    soft_rules_for_match = [(num, head, body_str) for (num, head, body_str, conf, penalty) in soft_rules]

    if VERBOSE:
        print(f"\n[SOFT BFS - BEST-PROOF (conf-priority + penalty)] Goal: {goal}")
        print("-" * 40)

    pq = []
    tie = count()

    def push_state(current, remaining, path, depth, soft_cost, min_conf, penalty_sum):
        if depth > max_depth:
            return
        heapq.heappush(
            pq,
            (soft_cost, -min_conf, penalty_sum, depth, next(tie),
             current, remaining, path, min_conf, penalty_sum)
        )

    push_state(goal, [], [], 0, 0, 1.0, 0.0)

    best_seen = {}

    def dominated(current, remaining, soft_cost, min_conf, penalty_sum, depth):
        key = (current, tuple(remaining))
        new = (soft_cost, -min_conf, penalty_sum, depth)
        old = best_seen.get(key)
        if old is None:
            best_seen[key] = new
            return False

        if (old[0], old[1], old[2]) < (new[0], new[1], new[2]):
            return True
        if (old[0], old[1], old[2]) == (new[0], new[1], new[2]) and old[3] <= new[3]:
            return True

        best_seen[key] = new
        return False

    def make_success_result(final_path, final_soft_cost, final_min_conf, final_depth, final_penalty_sum):
        used_soft = []
        for step in final_path:
            if step.startswith("SoftFact"):
                parts = step.split()
                if len(parts) >= 2:
                    try:
                        num = int(parts[1])
                    except ValueError:
                        num = None
                    used_soft.append(("fact", num))
            elif step.startswith("SoftRule"):
                parts = step.split()
                if len(parts) >= 2:
                    try:
                        num = int(parts[1])
                    except ValueError:
                        num = None
                    used_soft.append(("rule", num))

        return {
            "success": True,
            "proof_path": final_path,
            "used_soft_clauses": used_soft,
            "soft_cost": final_soft_cost,
            "min_conf": final_min_conf if final_soft_cost > 0 else None,
            "penalty_sum": final_penalty_sum,
            "depth": final_depth,
        }

    solutions = []

    def maybe_add_solution(res, soft_cost, min_conf, penalty_sum):
        solutions.append((soft_cost, -min_conf, penalty_sum, res))
        solutions.sort(key=lambda x: (x[0], x[1], x[2]))
        if len(solutions) > max_solutions:
            solutions.pop()

    while pq:
        soft_cost, neg_min_conf, penalty_sum, depth, _, current, remaining, path, min_conf, penalty_sum = heapq.heappop(pq)

        if VERBOSE:
            print(f"Depth {depth}: {current}")
            print(f"  Priority key: (soft_cost={soft_cost}, -min_conf={neg_min_conf:.3f}, penalty={penalty_sum:.3f}, depth={depth})")
            if remaining:
                print(f"  Remaining: {remaining}")

        if dominated(current, remaining, soft_cost, min_conf, penalty_sum, depth):
            continue

        # 1) HARD facts: exact match
        for num, fact in hard_facts:
            if check_exact_match(current, fact):
                if VERBOSE:
                    print(f"  ✓ Hard Fact {num} matches exactly: {fact}")
                new_path = path + [f"HardFact {num}"]

                if not remaining:
                    res = make_success_result(new_path, soft_cost, min_conf, depth + 1, penalty_sum)
                    maybe_add_solution(res, soft_cost, min_conf, penalty_sum)
                    break

                push_state(remaining[0], remaining[1:], new_path, depth + 1, soft_cost, min_conf, penalty_sum)
                break

        # 2) HARD facts: unification
        for num, fact in hard_facts:
            bindings = unify_with_fact(current, fact)
            if bindings is None:
                continue

            if VERBOSE:
                print(f"  ✓ Hard Fact {num} unifies: {fact}")
                print(f"    Bindings: {bindings}")

            instantiated = apply_bindings(remaining, bindings)
            new_path = path + [f"HardFact {num}"]

            if not instantiated:
                res = make_success_result(new_path, soft_cost, min_conf, depth + 1, penalty_sum)
                maybe_add_solution(res, soft_cost, min_conf, penalty_sum)
                continue

            push_state(instantiated[0], instantiated[1:], new_path, depth + 1, soft_cost, min_conf, penalty_sum)

        # 3) HARD rules
        matching_hard_rules = find_matching_rules_only(current, hard_rules)
        if VERBOSE and matching_hard_rules:
            print(f"  Matching hard rules: {matching_hard_rules}")

        for rule_num in matching_hard_rules:
            for num, head, body in hard_rules:
                if num != rule_num:
                    continue

                subgoals = get_subgoals(current, head, body)
                if not subgoals:
                    continue

                if VERBOSE:
                    print(f"  Hard Rule {num}: {head} :- {body}")
                    print(f"    → {subgoals}")

                all_goals = subgoals + remaining
                push_state(all_goals[0], all_goals[1:], path + [f"HardRule {num}"], depth + 1,
                           soft_cost, min_conf, penalty_sum)
                break

        # 4) SOFT facts
        for s_num, s_atom, s_conf, s_penalty in soft_facts:
            if max_soft is not None and soft_cost >= max_soft:
                break

            bindings = unify_with_fact(current, s_atom)
            if bindings is None:
                continue

            new_soft_cost = soft_cost + 1
            new_min_conf = min(min_conf, s_conf)
            new_penalty_sum = penalty_sum + float(s_penalty)

            if VERBOSE:
                print(f"  ✓ Soft Fact {s_num} unifies: {s_atom}")
                print(f"    Bindings: {bindings}, conf={s_conf:.3f}, penalty={float(s_penalty):.3f}")
                print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}, new penalty: {new_penalty_sum:.3f}")

            instantiated = apply_bindings(remaining, bindings)
            new_path = path + [f"SoftFact {s_num} (conf={s_conf:.3f},pen={float(s_penalty):.3f})"]

            if not instantiated:
                res = make_success_result(new_path, new_soft_cost, new_min_conf, depth + 1, new_penalty_sum)
                maybe_add_solution(res, new_soft_cost, new_min_conf, new_penalty_sum)
                continue

            push_state(instantiated[0], instantiated[1:], new_path, depth + 1,
                       new_soft_cost, new_min_conf, new_penalty_sum)

        # 5) SOFT rules (optional)
        matching_soft_rules = find_matching_rules_only(current, soft_rules_for_match)
        if VERBOSE and matching_soft_rules:
            print(f"  Matching soft rules: {matching_soft_rules}")

        for rule_num in matching_soft_rules:
            if max_soft is not None and soft_cost >= max_soft:
                break

            for s_num, s_head, s_body_str, s_conf, s_penalty in soft_rules:
                if s_num != rule_num:
                    continue

                subgoals = get_subgoals(current, s_head, s_body_str)
                if not subgoals:
                    continue

                new_soft_cost = soft_cost + 1
                new_min_conf = min(min_conf, s_conf)
                new_penalty_sum = penalty_sum + float(s_penalty)

                if VERBOSE:
                    print(f"  Soft Rule {s_num}: {s_head} :- {s_body_str}")
                    print(f"    → {subgoals}, conf={s_conf:.3f}, penalty={float(s_penalty):.3f}")
                    print(f"    New soft cost: {new_soft_cost}, new min_conf: {new_min_conf:.3f}, new penalty: {new_penalty_sum:.3f}")

                all_goals = subgoals + remaining
                new_path = path + [f"SoftRule {s_num} (conf={s_conf:.3f},pen={float(s_penalty):.3f})"]

                push_state(all_goals[0], all_goals[1:], new_path, depth + 1,
                           new_soft_cost, new_min_conf, new_penalty_sum)
                break

    if solutions:
        return solutions[0][3]

    if VERBOSE:
        print("✗ PRIORITY SOFT-BFS FAILED (no proof found even with soft KB)")
    return {
        "success": False,
        "proof_path": [],
        "used_soft_clauses": [],
        "soft_cost": None,
        "min_conf": None,
        "penalty_sum": None,
        "depth": None,
    }


# ============================================================
# Orchestration (paper-friendly metrics + final_proof_path fix)
# ============================================================

def solve_with_background(
    goal: str,
    kb: str,
    max_depth: Optional[int] = None,
    max_soft=None,
    hard_result=None,
):
    if max_depth is None:
        max_depth = config.DEFAULT_MAX_DEPTH
        
    predicate_comments = parse_kb_predicate_comments(kb)

    if VERBOSE:
        print("\n========================================")
        print(f"SOLVE WITH BACKGROUND: {goal}")
        print("========================================\n")

    if hard_result is None:
        if VERBOSE:
            print(">>> Phase 1: Hard-KB BFS (bfs_prolog_collect)")
        hard_result = bfs_prolog_collect(goal, kb, max_depth=max_depth)
        if VERBOSE:
            print("Hard-KB result:", hard_result)
    else:
        if VERBOSE:
            print(">>> Phase 1: Hard-KB BFS result already computed, reusing it.")
            print("Hard-KB result:", hard_result)

    if hard_result.get("success"):
        if VERBOSE:
            print("\n>>> Result: HARD_SUCCESS (no background hypotheses needed)\n")
        return {
            "status": "HARD_SUCCESS",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": [],
            "final_proof_path": hard_result.get("proof_path", []),
            "paper_metrics": {
                "hard_expansions": hard_result.get("metrics", {}).get("hard_expansions"),
                "blocking_atom": None,
                "llm_hypotheses_considered": 0,
                "soft_cost": 0,
                "min_conf": None,
            }
        }

    unresolved_atoms = hard_result.get("unresolved_atoms", set())
    if not unresolved_atoms:
        if VERBOSE:
            print("\nNo unresolved atoms to explain; cannot generate hypotheses.")
            print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": [],
            "final_proof_path": [],
            "paper_metrics": {
                "hard_expansions": hard_result.get("metrics", {}).get("hard_expansions"),
                "blocking_atom": None,
                "llm_hypotheses_considered": 0,
                "soft_cost": None,
                "min_conf": None,
            }
        }

    if VERBOSE:
        print("\n>>> Phase 2: Generate background hypotheses (FAST)")
        print("Unresolved atoms (minimal):", unresolved_atoms)

    hypotheses = generate_background_hypotheses_fast(
        goal=goal,
        kb=kb,
        hard_result=hard_result,
        predicate_comments=predicate_comments,
        max_atoms=config.HYP_MAX_ATOMS,
        max_hyp_per_atom=config.HYP_MAX_HYP_PER_ATOM,
        prompt_fact_limit=config.HYP_PROMPT_FACT_LIMIT,
    ) or []

    if not hypotheses:
        if VERBOSE:
            print("Hypotheses returned by LLM: []")
            print("\nLLM returned NO hypotheses; cannot build soft KB.")
            print(">>> Result: FAILURE\n")
        return {
            "status": "FAILURE",
            "hard_result": hard_result,
            "soft_result": None,
            "hypotheses": [],
            "final_proof_path": [],
            "paper_metrics": {
                "hard_expansions": hard_result.get("metrics", {}).get("hard_expansions"),
                "blocking_atom": next(iter(unresolved_atoms)) if unresolved_atoms else None,
                "llm_hypotheses_considered": 0,
                "soft_cost": None,
                "min_conf": None,
            }
        }

    if VERBOSE:
        print("Hypotheses returned by LLM:")
        for h in hypotheses:
            print("  - Clause:", h.get("clause"),
                  "| Conf:", h.get("confidence"),
                  "| From atom:", h.get("from_atom"),
                  "| Shortcut:", h.get("is_shortcut"),
                  "| UnknownStation:", h.get("unknown_station"))

        print("\n>>> Phase 3: Attach hypotheses to soft KB")

    soft_kb = attach_hypotheses_to_kb(kb, hypotheses)

    if VERBOSE:
        print("Soft KB facts:", soft_kb.get("facts", []))
        print("Soft KB rules:", soft_kb.get("rules", []))
        print("\n>>> Phase 4: Soft BFS (bfs_prolog_metro_soft)")

    soft_result = bfs_prolog_metro_soft(
        goal=goal,
        kb=kb,
        soft_kb=soft_kb,
        max_depth=max_depth,
        max_soft=max_soft,
    )

    if VERBOSE:
        print("Soft-BFS result:", soft_result)

    if soft_result.get("success"):
        if VERBOSE:
            print("\n>>> Result: SOFT_SUCCESS (proof found using background hypotheses)\n")
        final_path = soft_result.get("proof_path", [])
        return {
            "status": "SOFT_SUCCESS",
            "hard_result": hard_result,
            "soft_result": soft_result,
            "hypotheses": hypotheses,
            "final_proof_path": final_path,  # FIX: previously missing in your main print
            "paper_metrics": {
                "hard_expansions": hard_result.get("metrics", {}).get("hard_expansions"),
                "blocking_atom": hard_result.get("metrics", {}).get("blocking_atom") or (next(iter(unresolved_atoms)) if unresolved_atoms else None),
                "llm_hypotheses_considered": len(hypotheses),
                "llm_hypotheses_injected": len(soft_kb.get("facts", [])) + len(soft_kb.get("rules", [])),
                "soft_cost": soft_result.get("soft_cost"),
                "min_conf": soft_result.get("min_conf"),
                "penalty_sum": soft_result.get("penalty_sum"),
                "depth": soft_result.get("depth"),
            }
        }

    if VERBOSE:
        print("\n>>> Result: SOFT_FAILURE (no proof even with background hypotheses)\n")

    return {
        "status": "SOFT_FAILURE",
        "hard_result": hard_result,
        "soft_result": soft_result,
        "hypotheses": hypotheses,
        "final_proof_path": [],
        "paper_metrics": {
            "hard_expansions": hard_result.get("metrics", {}).get("hard_expansions"),
            "blocking_atom": hard_result.get("metrics", {}).get("blocking_atom") or (next(iter(unresolved_atoms)) if unresolved_atoms else None),
            "llm_hypotheses_considered": len(hypotheses),
            "llm_hypotheses_injected": len(soft_kb.get("facts", [])) + len(soft_kb.get("rules", [])),
            "soft_cost": None,
            "min_conf": None,
        }
    }


def omit_facts_from_kb(kb: str, omit_numbers):
    omit_numbers = set(omit_numbers)
    new_lines = []

    for line in kb.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.match(r"^(\d+)\.\s*(.+)$", line)
        if not m:
            continue

        num = int(m.group(1))
        if num in omit_numbers:
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


# ============================================================
# Natural language to Prolog (unchanged; still uses safe JSON)
# ============================================================

def nl_kb_to_prolog_kb(nl_kb_text: str, start_index: int = 1) -> list[str]:
    nl_kb_text = (nl_kb_text or "").strip()
    if not nl_kb_text:
        return []

    prompt = f"""
You are a Prolog formalization assistant.

The user will give you a natural-language description of a small domain,
including objects, relationships, and logical rules.

Your job is to convert that description into a set of Prolog clauses
(facts and rules).

Guidelines:
- Use lowercase atoms for concrete entities (e.g. union_square, times_square,
  grand_central, bryant_park).
- Use uppercase identifiers for variables (e.g. X, Y, Z).
- Choose predicate names that are short, descriptive, and consistent,
  for example: connected/2, reachable/2, located_in/2, etc.
- A fact must look like:
    connected(times_square, bryant_park).
- A rule must look like:
    reachable(X, Z) :- connected(X, Y), reachable(Y, Z).
- Every clause MUST end with a single period '.'.
- Do NOT include any line numbers in your output clauses.
- Do NOT add explanations or comments in the Prolog code.

Here is the NATURAL LANGUAGE description of the knowledge base:

\"\"\"{nl_kb_text}\"\"\"

Respond ONLY in this JSON format (and nothing else):

{{
  "clauses": [
    {{
      "clause": "connected(union_square, times_square)."
    }},
    {{
      "clause": "reachable(X, Y) :- connected(X, Y)."
    }}
  ]
}}
""".strip()

    raw = ask_llm(prompt).strip()
    try:
        data = json.loads(extract_first_json(raw))
    except Exception as e:
        print("[nl_kb_to_prolog_kb] JSON parse error:", e)
        print("Raw LLM output:", raw[:800])
        return []

    raw_clauses = data.get("clauses", [])
    if not isinstance(raw_clauses, list):
        print("[nl_kb_to_prolog_kb] 'clauses' field is not a list:", raw_clauses)
        return []

    cleaned_clauses = []
    for item in raw_clauses:
        if isinstance(item, str):
            clause = item.strip()
        elif isinstance(item, dict):
            clause = (item.get("clause") or "").strip()
        else:
            continue

        if not clause:
            continue

        m_num = re.match(r"^\s*(\d+)\.\s*(.+)$", clause)
        if m_num:
            clause = m_num.group(2).strip()

        clause = clause.rstrip()
        if not clause.endswith("."):
            clause = clause + "."
        else:
            clause = re.sub(r"\.+$", ".", clause)

        body_str = clause[:-1].strip()
        if ":-" in body_str:
            head_part, _ = body_str.split(":-", 1)
            head = head_part.strip()
        else:
            head = body_str

        parsed_head = parse_predicate(head)
        if parsed_head is None:
            print("[nl_kb_to_prolog_kb] Discarding unparsable clause:", clause)
            continue

        cleaned_clauses.append(clause)

    numbered_clauses = []
    next_num = start_index
    for clause in cleaned_clauses:
        numbered_clauses.append(f"{next_num}. {clause}")
        next_num += 1

    return numbered_clauses


def generate_background_hypotheses_fast(
    goal: str,
    kb: str,
    hard_result: dict,
    predicate_comments: dict,
    max_atoms: int = 1,
    max_hyp_per_atom: int = 2,
    prompt_fact_limit: int = 25,
):
    """
    FAST background hypothesis generation (single-call + cache), with:
      - Safer JSON extraction
      - Deterministic atom selection
      - Treat constants in goal + unresolved atoms as "known stations"
      - Robust JSON-only LLM call via llm_json_only(prompt, repair_schema)
      - Hard reject shortcuts (adjacent-only semantics)
      - Hard reject duplicates already in hard KB
      - Record correctness: force from_atom to be one of the requested atoms
      - FIX: replace overly-strict unification-only filter with "blocking-compatible" filter:
             accept hypotheses that either unify with the blocking atom OR share an endpoint
             with the blocking atom (handles metro case where blocking is ground and missing edge
             is adjacent to one endpoint).
      - Minimal, defensible audit hooks (internal)

    Returns: list[dict] hypotheses:
        {
          "clause": "connected(u, v).",
          "confidence": 0.0..1.0,
          "from_atom": "<requested atom>",
          "is_shortcut": bool,
          "shortcut_len": Optional[int],
          "unknown_station": bool,
          "unifies_with_blocking": bool,
          "blocking_atom": str,
          "blocking_compatible": bool
        }
    """

    # -----------------------------
    # Local helper: extract constants from a term like pred(a, B, c)
    # -----------------------------
    def _extract_constants_from_term(term: str) -> set:
        out = set()
        p = parse_predicate((term or "").strip().rstrip("."))
        if not p:
            return out
        _, args = p
        for a in args:
            a = a.strip()
            if not a:
                continue
            if is_variable(a):
                continue
            if re.match(r"^[a-z_][a-zA-Z0-9_]*$", a):
                out.add(a)
        return out

    # -----------------------------
    # Local helper: unify two predicate strings (very small, Prolog-ish)
    # - Returns True if unifiable under standard first-order unification
    #   restricted to constants + variables (no function symbols).
    # -----------------------------
    def _is_var(tok: str) -> bool:
        return is_variable(tok)

    def _unifiable(atom_a: str, atom_b: str) -> bool:
        pa = parse_predicate((atom_a or "").strip().rstrip("."))
        pb = parse_predicate((atom_b or "").strip().rstrip("."))
        if not (pa and pb):
            return False
        fa, aa = pa
        fb, ab = pb
        if fa != fb or len(aa) != len(ab):
            return False

        subst = {}  # var -> term

        def deref(t: str) -> str:
            while _is_var(t) and t in subst:
                t = subst[t]
            return t

        for x0, y0 in zip(aa, ab):
            x = deref(x0.strip())
            y = deref(y0.strip())

            if x == y:
                continue

            x_is_var = _is_var(x)
            y_is_var = _is_var(y)

            if x_is_var and y_is_var:
                subst[x] = y
            elif x_is_var and not y_is_var:
                subst[x] = y
            elif not x_is_var and y_is_var:
                subst[y] = x
            else:
                return False

        return True

    # -----------------------------
    # FIX: blocking-compatibility check
    # - If blocking atom is ground connected(a,b), accept any candidate connected(u,v)
    #   that touches either endpoint (u==a or v==a or u==b or v==b).
    # - If blocking has vars, fall back to unification.
    # -----------------------------
    def _shares_endpoint_with_blocking(u: str, v: str, blocking_atom: str) -> bool:
        pb = parse_predicate((blocking_atom or "").strip().rstrip("."))
        if not pb:
            return False
        fb, ab = pb
        if fb != "connected" or len(ab) != 2:
            return False

        b1, b2 = ab[0].strip(), ab[1].strip()

        # if blocking includes variables, unification is the correct notion
        if _is_var(b1) or _is_var(b2):
            return _unifiable(f"connected({u}, {v})", blocking_atom)

        # ground blocking: accept edges incident to either endpoint
        return (u == b1) or (v == b1) or (u == b2) or (v == b2)

    unresolved_atoms = hard_result.get("unresolved_atoms", set()) if hard_result else set()
    if not unresolved_atoms:
        return []

    # Determine the "blocking atom"
    blocking_atom = None
    try:
        blocking_atom = hard_result.get("metrics", {}).get("blocking_atom", None)
    except Exception:
        blocking_atom = None
    if not blocking_atom and unresolved_atoms:
        blocking_atom = sorted(list(unresolved_atoms))[0]
    if not blocking_atom:
        return []

    # Build hard KB adjacency + station set from KB facts
    hard_adj, hard_edges, stations, connected_facts = _extract_connected_facts_and_stations(kb)

    # Treat goal + unresolved constants as “known”
    stations_aug = set(stations)
    stations_aug |= _extract_constants_from_term(goal)
    for a in unresolved_atoms:
        stations_aug |= _extract_constants_from_term(a)

    # Choose which atoms to ask the LLM about
    preferred_atom = _infer_preferred_atom_for_chain(goal, hard_adj)
    atoms = _select_atoms_for_bg(unresolved_atoms, preferred_atom, max_atoms=max_atoms)

    if not atoms:
        print("[generate_background_hypotheses_fast] No suitable ground atoms.")
        return []

    # Cache
    kb_sig = _kb_signature_for_bg(kb)
    cache_key = (kb_sig, goal, tuple(atoms), max_hyp_per_atom, prompt_fact_limit, blocking_atom)
    if cache_key in _BG_HYP_CACHE:
        return _BG_HYP_CACHE[cache_key]

    # Prompt snapshot
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

    # Repair schema pins the key to our requested atom
    repair_schema = f"""{{
  "by_atom": {{
    "{atoms[0]}": [
      {{"clause":"connected(grand_central, bryant_park).","confidence":0.95}},
      {{"clause":"connected(42nd_street, bryant_park).","confidence":0.55}}
    ]
  }}
}}"""

    raw = llm_json_only(prompt, repair_schema).strip()
    if DEBUG:
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

    # ============================================================
    # Force 'from_atom' correctness + robust handling of LLM keys
    # ============================================================
    requested_atoms = list(atoms)
    requested_set = set(requested_atoms)
    single_target = requested_atoms[0] if len(requested_atoms) == 1 else None

    filtered_by_atom = {}
    for k, v in by_atom.items():
        if not isinstance(v, list):
            continue
        if k in requested_set:
            filtered_by_atom[k] = v
        elif single_target is not None:
            filtered_by_atom.setdefault(single_target, []).extend(v)

    by_atom = filtered_by_atom
    if not by_atom:
        _BG_HYP_CACHE[cache_key] = []
        return []

    # ------------------------------------------------------------
    # Normalization + validator helpers
    # ------------------------------------------------------------
    def norm_clause(cl: str):
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

    # ------------------------------------------------------------
    # Build proposal records
    # ------------------------------------------------------------
    accepted = []
    dedup_best = {}  # clause -> best record

    for atom_key, proposals in by_atom.items():
        forced_from_atom = atom_key

        for item in proposals[:max_hyp_per_atom]:
            if not isinstance(item, dict):
                continue

            raw_clause = (item.get("clause", "") or "").strip()
            clause = norm_clause(raw_clause)
            if clause is None:
                continue

            try:
                conf = float(item.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            atom_str = clause.rstrip(".").strip()
            p = parse_predicate(atom_str)
            u, v = p[1][0].strip(), p[1][1].strip()

            # Reject duplicates already in hard KB
            if (u, v) in hard_edges:
                continue

            # Reject shortcut edges (adjacent-only semantics)
            is_shortcut, d = shortcut_meta(u, v)
            if is_shortcut:
                continue

            unknown_station = bool(stations_aug) and (u not in stations_aug or v not in stations_aug)

            # ========================================================
            # FIX: keep hypotheses that are blocking-compatible:
            # - share endpoint with blocking atom (ground case), OR
            # - unify with blocking atom (var case)
            # ========================================================
            unifies_with_blocking = _unifiable(atom_str, blocking_atom)
            blocking_compatible = _shares_endpoint_with_blocking(u, v, blocking_atom)

            if not (blocking_compatible or unifies_with_blocking):
                continue

            rec = {
                "clause": clause,
                "confidence": conf,
                "from_atom": forced_from_atom,
                "is_shortcut": False,
                "shortcut_len": None,
                "unknown_station": unknown_station,
                "unifies_with_blocking": unifies_with_blocking,
                "blocking_atom": blocking_atom,
                "blocking_compatible": blocking_compatible,
            }
            accepted.append(rec)

    # Dedup: keep highest confidence per identical clause
    for h in accepted:
        key = h["clause"]
        if key not in dedup_best or h["confidence"] > dedup_best[key]["confidence"]:
            dedup_best[key] = h

    result = list(dedup_best.values())
    _BG_HYP_CACHE[cache_key] = result
    return result


# ============================================================
# Main (compatible; now prints paper_metrics + final_proof_path)
# ============================================================

if __name__ == "__main__":
    kb = """
    1. connected(union_square, 14th_street). # connected/2 = ADJACENT STOPS ONLY. Use only when two stations are immediate neighbors on the same line. Do NOT add shortcut edges that skip intermediate stations.
    2. connected(14th_street, 23rd_street).
    3. connected(23rd_street, 34th_street).
    4. connected(34th_street, times_square).
    5. connected(times_square, 42nd_street).
    6. connected(42nd_street, grand_central).
    7. connected(grand_central, bryant_park).
    8. reachable(X, Y) :- connected(X, Y). # reachable/2 = PATH EXISTS. One-hop reachable comes only from connected/2.
    9. reachable(X, Z) :- connected(X, Y), reachable(Y, Z). # reachable/2 is the transitive closure of connected/2 (multi-hop path following only adjacent edges).
    """

    print("===== FULL METRO KB =====")
    print(kb)
    print("====================================\n")

    # Omit fact 7 to force background reasoning
    kb_missing_fact = omit_facts_from_kb(kb, omit_numbers={7})

    print("===== METRO KB WITH FACT REMOVED =====")
    print(kb_missing_fact)
    print("========================================\n")

    test_goal = "reachable(union_square, bryant_park)"

    print("==============================")
    print(f"TEST QUERY: {test_goal}")
    print("==============================\n")

    print(">>> Running bfs_prolog_collect (hard-KB BFS)...")
    collect_result = bfs_prolog_collect(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=20
    )
    print("Collect Result:", collect_result)
    print("\n----------------------------------------\n")

    print(">>> Running solve_with_background (full pipeline, reusing hard result)...")
    bg_result = solve_with_background(
        goal=test_goal,
        kb=kb_missing_fact,
        max_depth=20,
        hard_result=collect_result,
    )
    print("Solve-with-background Result:")
    print(bg_result)

    print("\n===== PAPER METRICS (results section) =====")
    print(bg_result.get("paper_metrics", {}))
    print("==========================================\n")

    if bg_result.get("status") == "SOFT_SUCCESS":
        print("\n===== FULL PROOF PATH (ROOT GOAL PROVEN) =====")
        for step in bg_result.get("final_proof_path", []):
            print(step)
        print("============================================\n")
