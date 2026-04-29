"""Prolog to logic formula parsing utilities."""

import re
from typing import Optional
from typing import Tuple

from prolog.prolog_utils import is_variable

from logic.logic import (
    Pred,
    Var,
    Const,
    AtomicFormula,
    Formula,
    Conjunction,
)


def parse_prolog_to_formula(term: str) -> AtomicFormula:
    """Parse a Prolog term string to an AtomicFormula."""
    term = term.strip().rstrip(".")
    functor, args = parse_predicate(term)
    pred = Pred(functor, len(args))
    
    term_args = []
    for arg in args:
        arg = arg.strip()
        if is_variable(arg):
            term_args.append(Var(arg))
        else:
            term_args.append(Const(arg))
    
    return pred(*term_args)


def parse_body_to_formula(body_str: str) -> Optional[Formula]:
    """Parse a Prolog rule body string to a Formula."""
    body_str = body_str.strip()
    if not body_str:
        return None
    
    atoms = split_body_atoms(body_str)
    assert atoms is not None, "I think that this assert is redundant."

    formulas = [parse_prolog_to_formula(atom) for atom in atoms]
    
    if len(formulas) == 1:
        return formulas[0]
    
    result = formulas[0]
    for f in formulas[1:]:
        result = Conjunction(result, f)
    
    return result


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


def parse_predicate(term: str):
    """
    Parse a simple Prolog predicate of the form:
        functor(arg1, arg2, ...)
    Returns (functor: str, args: list[str]) or None if parsing fails.
    """
    term = term.strip().rstrip(".")
    m = re.match(r"^([a-z_][a-zA-Z0-9_]*)\((.*)\)$", term)
    if not m:
        raise SyntaxError(f"Cannot parse Prolog term {term}.")

    functor = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        args = []
    else:
        args = [a.strip() for a in args_raw.split(",")]
    return functor, args


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


def split_head_and_body(clause: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Splits the formula into head and body.
    """
    clause = (clause or "").strip()
    if not clause:
        return None, None
    if not clause.endswith("."):
        clause += "."

    body = clause[:-1].strip()
    if ":-" in body:
        head, body_part = body.split(":-", 1)
        head = head.strip()
        body_part = body_part.strip()
        try:
            parse_predicate(head)
            for atom in split_body_atoms(body_part):
                parse_predicate(atom.strip())
        except Exception:
            return None, None
        return head, body_part

    try:
        parse_predicate(body)
    except Exception:
        return None, None
    return body, None