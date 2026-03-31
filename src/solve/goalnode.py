from dataclasses import dataclass
from typing import List

import config
from logic.logic import AtomicFormula, Conjunction, Formula, Term, Var
from prolog.knowledge_base import SoftKnowledgeBase
from prolog.prolog_command import SoftFact, SoftRule


@dataclass
class GoalNode:
    def __init__(self, formulas: List[AtomicFormula], depth: int, confidence: float):
        self._formulas = formulas
        self._proved = [0.0 for _ in formulas]
        self.depth = depth
        self.confidence = confidence

    def _aggregated_confidence(self, clause_confidence: float) -> float:
        """Aggregate the node confidence with a clause confidence."""
        return config.PATH_AGGREGATION_FUNCTION(clause_confidence, self.confidence)

    def mark_proved_fact(self, fact: SoftFact, min_confidence: float = 0.0) -> float:
        """Mark all occurrences of ``fact.formula`` as proved."""
        found = 0.0
        aggregated_confidence = self._aggregated_confidence(fact.confidence)
        if aggregated_confidence < min_confidence:
            return 0.0
        for idx, formula in enumerate(self._formulas):
            if formula == fact.formula:
                self._proved[idx] = aggregated_confidence
                found = aggregated_confidence
        return found

    def mark_proved_facts(self, soft_kb: SoftKnowledgeBase, min_confidence: float = 0.0) -> float:
        """Mark all formulas proved by matching facts from a soft KB."""
        found = 0.0
        for fact in soft_kb.facts:
            found = max(found, self.mark_proved_fact(fact, min_confidence=min_confidence))
        return found

    def _unify_formula_with_fact(self, formula: AtomicFormula, fact: SoftFact) -> dict[Term, Term] | None:
        """Unify one formula with a soft fact and return the induced substitution."""
        fact_formula = fact.formula

        if formula.pred != fact_formula.pred or len(formula.args) != len(fact_formula.args):
            return None

        bindings: dict[Term, Term] = {}
        for goal_arg, fact_arg in zip(formula.args, fact_formula.args):
            goal_is_var = isinstance(goal_arg, Var)
            fact_is_var = isinstance(fact_arg, Var)

            if not goal_is_var and not fact_is_var:
                if goal_arg != fact_arg:
                    return None
                continue

            if goal_is_var and not fact_is_var:
                if goal_arg in bindings and bindings[goal_arg] != fact_arg:
                    return None
                bindings[goal_arg] = fact_arg
                continue

            if not goal_is_var and fact_is_var:
                if fact_arg in bindings and bindings[fact_arg] != goal_arg:
                    return None
                bindings[fact_arg] = goal_arg
                continue

            if goal_arg in bindings and fact_arg in bindings:
                if bindings[goal_arg] != bindings[fact_arg]:
                    return None
            elif goal_arg in bindings:
                bindings[fact_arg] = bindings[goal_arg]
            elif fact_arg in bindings:
                bindings[goal_arg] = bindings[fact_arg]
            else:
                bindings[fact_arg] = goal_arg

        return bindings

    def _split_body_formulas(self, formula: Formula | None) -> List[AtomicFormula]:
        """Flatten a rule body into an ordered list of atomic formulas."""
        if formula is None:
            return []
        if isinstance(formula, AtomicFormula):
            return [formula]
        if isinstance(formula, Conjunction):
            return self._split_body_formulas(formula.left_formula) + self._split_body_formulas(formula.right_formula)
        raise TypeError("Rule body can contain only conjunctions of atomic formulas.")

    def _standardize_apart_rule(
        self,
        rule: SoftRule,
        formula_idx: int,
    ) -> tuple[AtomicFormula, List[AtomicFormula]]:
        """Rename rule variables to avoid collisions with variables already present in the goal node."""
        body_formulas = self._split_body_formulas(rule.body_formula)
        all_rule_vars = set(rule.head_formula.vars())
        for body_formula in body_formulas:
            all_rule_vars.update(body_formula.vars())

        renaming = {
            variable: Var(f"__r{rule.num}_d{self.depth}_i{formula_idx}_{variable.name}")
            for variable in all_rule_vars
        }
        standardized_head = rule.head_formula.substitute(renaming)
        standardized_body = [body_formula.substitute(renaming) for body_formula in body_formulas]
        return standardized_head, standardized_body

    def unify_soft_fact(self, fact: SoftFact, min_confidence: float = 0.0) -> List["GoalNode"]:
        """Create one successor node for each formula with free variables that unifies with ``fact``."""
        new_nodes: List[GoalNode] = []
        aggregated_confidence = self._aggregated_confidence(fact.confidence)
        if aggregated_confidence < min_confidence:
            return new_nodes

        for idx, formula in enumerate(self._formulas):
            if not formula.vars():
                continue

            bindings = self._unify_formula_with_fact(formula, fact)
            if bindings is None:
                continue

            new_formulas = [current_formula.substitute(bindings) for current_formula in self._formulas]
            new_node = GoalNode(new_formulas, self.depth + 1, aggregated_confidence)
            new_node._proved = self._proved[:]
            new_node._proved[idx] = aggregated_confidence
            new_nodes.append(new_node)

        return new_nodes

    def unify_soft_kb(self, soft_kb: SoftKnowledgeBase, min_confidence: float = 0.0) -> List["GoalNode"]:
        """Return all successor goal nodes produced by unifying with facts from a soft KB."""
        new_nodes: List[GoalNode] = []
        for fact in soft_kb.facts:
            new_nodes.extend(self.unify_soft_fact(fact, min_confidence=min_confidence))
        return new_nodes

    def unify_formula_with_soft_rule(
        self,
        formula: AtomicFormula,
        rule: SoftRule,
        min_confidence: float = 0.0,
    ) -> "GoalNode|None":
        """Replace a single formula with the instantiated body of a unifying soft rule."""
        try:
            formula_idx = self._formulas.index(formula)
        except ValueError:
            return None

        # no point in unifying facts with those from the KnowledgeBase
        if self._proved[formula_idx] > 0:
            return None

        standardized_head, standardized_body = self._standardize_apart_rule(rule, formula_idx)
        bindings = self._unify_formula_with_fact(formula, SoftFact(rule.num, standardized_head, rule.confidence))
        if bindings is None:
            return None

        body_formulas = [body_formula.substitute(bindings) for body_formula in standardized_body]
        new_formulas = self._formulas[:formula_idx] + body_formulas + self._formulas[formula_idx + 1:]

        aggregated_confidence = self._aggregated_confidence(rule.confidence)
        if aggregated_confidence < min_confidence:
            return None
        new_node = GoalNode(new_formulas, self.depth + 1, aggregated_confidence)
        new_node._proved = (
            self._proved[:formula_idx]
            + [0.0 for _ in body_formulas]
            + self._proved[formula_idx + 1:]
        )
        return new_node

    def unify_soft_rule(self, rule: SoftRule, min_confidence: float = 0.0) -> List["GoalNode"]:
        """Return all successor goal nodes produced by unifying this node with one soft rule."""
        new_nodes: List[GoalNode] = []
        for formula in self._formulas:
            new_node = self.unify_formula_with_soft_rule(formula, rule, min_confidence=min_confidence)
            if new_node is not None:
                new_nodes.append(new_node)
        return new_nodes

    def unify_soft_rules(self, soft_kb: SoftKnowledgeBase, min_confidence: float = 0.0) -> List["GoalNode"]:
        """Return all successor goal nodes produced by unifying this node with rules from a soft KB."""
        new_nodes: List[GoalNode] = []
        for rule in soft_kb.rules:
            new_nodes.extend(self.unify_soft_rule(rule, min_confidence=min_confidence))
        return new_nodes

    def is_proven(self) -> float:
        """Return the aggregated proof confidence if all formulas are proved, else ``0.0``."""
        if not self._formulas:
            return self.confidence

        if any(proof_confidence <= 0.0 for proof_confidence in self._proved):
            return 0.0

        proof_confidence = self._proved[0]
        for current_confidence in self._proved[1:]:
            proof_confidence = config.PATH_AGGREGATION_FUNCTION(proof_confidence, current_confidence)
        return proof_confidence

    def unresolved_formulas(self) -> List[AtomicFormula]:
        """Return only formulas that are not proved yet."""
        return [
            formula
            for formula, proof_confidence in zip(self._formulas, self._proved)
            if proof_confidence <= 0.0
        ]

    def signature(self) -> tuple[str, ...]:
        """Return a canonical signature of the unresolved formulas.

        Variables are renamed consistently by first occurrence, so alpha-equivalent
        goals map to the same signature.
        """
        var_mapping: dict[Var, Var] = {}
        canonical_formulas: List[str] = []

        for formula in self.unresolved_formulas():
            for variable in formula.vars():
                if variable not in var_mapping:
                    var_mapping[variable] = Var(f"_G{len(var_mapping)}")
            canonical_formulas.append(str(formula.substitute(var_mapping)))

        return tuple(canonical_formulas)

    def __repr__(self) -> str:
        formulas = ", ".join(str(formula) for formula in self._formulas)
        proved = ", ".join(f"{confidence:.3f}" for confidence in self._proved)
        return (
            f"GoalNode(formulas=[{formulas}], proved=[{proved}], "
            f"depth={self.depth}, confidence={self.confidence:.3f}, "
            f"proof_confidence={self.is_proven():.3f})"
        )
