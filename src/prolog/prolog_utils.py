from typing import Tuple

from logic.logic import Var, Const, AtomicFormula



def is_variable(s: str) -> bool:
    """Prolog-ish variable check: starts with uppercase letter or '_'."""
    s = s.strip()
    return bool(s) and (s[0].isupper() or s[0] == "_")


def has_variables_and_constants(formula: AtomicFormula) -> Tuple[bool, bool]:
    has_variable = any(isinstance(arg, Var) for arg in formula.args)
    has_constant = any(isinstance(arg, Const) for arg in formula.args)
    return has_variable, has_constant


def is_mixed_atom(formula: AtomicFormula) -> bool:
    """Return whether the atom contains at least one variable and one constant."""
    has_variable, has_constant = has_variables_and_constants(formula)
    return has_variable and has_constant


def is_grounded_atom(formula: AtomicFormula) -> bool:
    has_variable, _ = has_variables_and_constants(formula)
    return not has_variable