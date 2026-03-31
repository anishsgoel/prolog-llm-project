"""Prolog parsing and unification utilities."""

from prolog.formula_parsing import split_body_atoms, parse_predicate
from prolog.prolog_utils import is_variable


def strip_inline_comment(s: str) -> str:
    return s.split("#", 1)[0].rstrip()


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


def is_ground_atom(atom: str) -> bool:
    """
    Groundness filter for metric collection: no variables in arg list.
    Keeps the solve logic unchanged; only affects what we log/hand to LLM.
    """
    p = parse_predicate(atom.strip().rstrip("."))
    if not p:
        return False
    _, args = p
    return all(not is_variable(a) for a in args)


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
