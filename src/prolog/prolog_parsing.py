"""Prolog to logic formula parsing utilities."""

from typing import Optional

from prolog_llm.prolog_utils import (
    parse_predicate,
    split_body_atoms,
    is_variable,
)

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
    parsed = parse_predicate(term)
    if parsed is None:
        raise ValueError(f"Cannot parse Prolog term: {term}")
    
    functor, args = parsed
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
    if not atoms:
        return None
    
    formulas = [parse_prolog_to_formula(atom) for atom in atoms]
    
    if len(formulas) == 1:
        return formulas[0]
    
    result = formulas[0]
    for f in formulas[1:]:
        result = Conjunction(result, f)
    
    return result
