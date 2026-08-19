#!/usr/bin/env python
"""Tests for the London benchmark generator.

Run with:
    python -m unittest discover -s tools -p "test_*.py"

Only the generator and the prompt-size filter are covered here; the network data
itself is checked by `python tools/build_london_benchmark.py verify`, which
diffs against a checked-in third-party edge list.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TOOLS_DIR.parent
for _p in (str(_TOOLS_DIR), str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_london_benchmark import (  # noqa: E402
    build_graph,
    build_tier,
    canonical_atom,
    check_express_skips,
    emit_kb,
    load_network,
    required_proof_depth,
)


class TestCanonicalAtom(unittest.TestCase):
    """The worked examples from the benchmark spec, verbatim."""

    CASES = {
        "King's Cross St Pancras": "kings_cross_st_pancras",
        "Shepherd's Bush": "shepherds_bush",
        "Shepherd's Bush Market": "shepherds_bush_market",
        "St James's Park": "st_jamess_park",
        "St John's Wood": "st_johns_wood",
        "St Paul's": "st_pauls",
        "Regent's Park": "regents_park",
        "Queen's Park": "queens_park",
        "Earl's Court": "earls_court",
        "Elephant & Castle": "elephant_and_castle",
        "Harrow & Wealdstone": "harrow_and_wealdstone",
        "Totteridge & Whetstone": "totteridge_and_whetstone",
        "Chalfont & Latimer": "chalfont_and_latimer",
        "Highbury & Islington": "highbury_and_islington",
        "Heathrow Terminals 2 & 3": "heathrow_terminals_2_and_3",
        "Heathrow Terminal 4": "heathrow_terminal_4",
        "Heathrow Terminal 5": "heathrow_terminal_5",
        "Harrow-on-the-Hill": "harrow_on_the_hill",
        "Bromley-by-Bow": "bromley_by_bow",
        "Kensington (Olympia)": "kensington_olympia",
        "Edgware Road (Bakerloo)": "edgware_road_bakerloo",
        "Edgware Road": "edgware_road",
        "Lambeth North": "lambeth_north",
    }

    def test_worked_examples(self):
        for name, expected in self.CASES.items():
            with self.subTest(name=name):
                self.assertEqual(canonical_atom(name), expected)

    def test_typographic_apostrophe_folds_to_ascii(self):
        self.assertEqual(canonical_atom("Earl’s Court"), "earls_court")

    def test_atoms_are_never_read_as_prolog_variables(self):
        """is_variable() treats a leading uppercase OR underscore as a variable."""
        from prolog.prolog_utils import is_variable

        for name in self.CASES:
            self.assertFalse(is_variable(canonical_atom(name)), name)

    def test_atoms_parse_as_predicate_arguments(self):
        from prolog.formula_parsing import parse_predicate

        for name in self.CASES:
            atom = canonical_atom(name)
            functor, args = parse_predicate(f"connected({atom}, oxford_circus)")
            self.assertEqual(functor, "connected")
            self.assertEqual(args[0], atom)


class TestNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = load_network()
        cls.graph, cls.suppressed = build_graph(cls.net)

    def test_station_and_edge_counts(self):
        self.assertEqual(len(self.graph.stations), 270)
        self.assertEqual(len(self.graph.edges), 306)

    def test_no_atom_collisions(self):
        atoms = [canonical_atom(s) for s in self.graph.stations]
        self.assertEqual(len(atoms), len(set(atoms)))

    def test_express_skips_absent_and_corridors_intact(self):
        self.assertEqual(check_express_skips(self.net, self.graph), [])

    def test_distinct_stations_stay_distinct(self):
        for a, b in [("Shepherd's Bush", "Shepherd's Bush Market"),
                     ("Edgware Road", "Edgware Road (Bakerloo)"),
                     ("Edgware", "Edgware Road")]:
            self.assertIn(a, self.graph.stations)
            self.assertIn(b, self.graph.stations)
            self.assertNotEqual(canonical_atom(a), canonical_atom(b))

    def test_closed_stations_absent(self):
        for closed in ("Aldwych", "Brompton Road", "Down Street", "York Road",
                       "British Museum", "Ongar", "North Weald", "Blake Hall"):
            self.assertNotIn(closed, self.graph.stations)

    def test_graph_is_connected(self):
        from build_london_benchmark import _connected_components
        self.assertEqual(len(_connected_components(self.graph)), 1)


class TestKbEmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        net = load_network()
        graph, _ = build_graph(net)
        cls.tier = build_tier(net, graph, "t1", net.tiers["t1"], net.zones)
        cls.text, cls.index = emit_kb(cls.tier)

    def test_both_directions_are_adjacent_and_consecutive(self):
        for entry in self.index:
            self.assertEqual(entry["fact_ba"], entry["fact_ab"] + 1)

    def test_rules_come_last_and_there_is_no_symmetry_rule(self):
        lines = [l for l in self.text.strip().split("\n") if l.strip()]
        self.assertIn("reachable(X, Y) :- connected(X, Y).", lines[-2])
        self.assertIn("reachable(X, Z) :- connected(X, Y), reachable(Y, Z).", lines[-1])
        self.assertNotIn("connected(X, Y) :- connected(Y, X)", self.text)

    def test_kb_parses_and_round_trips(self):
        from prolog.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(self.text)
        self.assertEqual(len(kb.facts), 2 * len(self.tier.graph.edges))
        self.assertEqual(len(kb.rules), 2)
        self.assertIn("connected/2", kb.predicate_comments)
        self.assertIn("reachable/2", kb.predicate_comments)

    def test_required_proof_depth_matches_measurements(self):
        # Measured on T1: hops 1..5 need max_depth 1, 3, 4, 6, 7.
        self.assertEqual([required_proof_depth(h) for h in range(1, 6)],
                         [1, 3, 4, 6, 7])


class TestSelectHardFacts(unittest.TestCase):
    """The §8.1 prompt-size filter."""

    def _facts(self, pairs):
        from prolog.prolog_command import SoftFact
        return [SoftFact(i + 1, f"connected({a}, {b})", 1.0)
                for i, (a, b) in enumerate(pairs)]

    def test_small_kb_is_passed_through_untouched(self):
        """Existing small configs must produce byte-identical prompts."""
        from solve.llm_search_guidance_policy import select_hard_facts

        facts = self._facts([("a", "b"), ("b", "c"), ("c", "d")])
        kept, withheld = select_hard_facts(facts, {"a"}, limit=400, hops=3)
        self.assertEqual(kept, facts)
        self.assertEqual(withheld, 0)

    def test_large_kb_is_restricted_to_the_goal_neighbourhood(self):
        from solve.llm_search_guidance_policy import select_hard_facts

        # a long chain a0-a1-...-a49; the goal only mentions a0
        chain = [(f"a{i}", f"a{i + 1}") for i in range(49)]
        facts = self._facts(chain)
        kept, withheld = select_hard_facts(facts, {"a0"}, limit=10, hops=2)
        self.assertEqual(len(kept), 2)
        self.assertEqual(withheld, len(facts) - 2)
        atoms = " ".join(f.atom for f in kept)
        self.assertIn("a0", atoms)
        self.assertNotIn("a40", atoms)

    def test_limit_keeps_the_nearest_facts(self):
        from solve.llm_search_guidance_policy import select_hard_facts

        chain = [(f"a{i}", f"a{i + 1}") for i in range(49)]
        facts = self._facts(chain)
        kept, withheld = select_hard_facts(facts, {"a0"}, limit=3, hops=40)
        self.assertEqual(len(kept), 3)
        self.assertEqual([f.atom for f in kept],
                         ["connected(a0, a1)", "connected(a1, a2)", "connected(a2, a3)"])
        self.assertEqual(withheld, len(facts) - 3)

    def test_no_seed_constants_still_returns_something_bounded(self):
        from solve.llm_search_guidance_policy import select_hard_facts

        chain = [(f"a{i}", f"a{i + 1}") for i in range(49)]
        facts = self._facts(chain)
        kept, withheld = select_hard_facts(facts, set(), limit=5, hops=3)
        self.assertLessEqual(len(kept), 5)
        self.assertEqual(withheld, len(facts) - len(kept))


def _import_batch_run():
    """Import batch_run, stubbing mlflow if it is not installed.

    ``_expand_config_patterns`` is pure argument handling and never touches
    mlflow, but batch_run imports it at module scope. Stubbing keeps this test
    meaningful on a machine without the tracking stack rather than silently
    skipping it.
    """
    import importlib.machinery
    import importlib.util
    import types

    if "mlflow" not in sys.modules:
        try:
            installed = importlib.util.find_spec("mlflow") is not None
        except (ImportError, ValueError):
            installed = False
        if not installed:
            stub = types.ModuleType("mlflow")
            # find_spec() raises ValueError on a sys.modules entry whose
            # __spec__ is None, which would break the second call.
            stub.__spec__ = importlib.machinery.ModuleSpec("mlflow", loader=None)
            sys.modules["mlflow"] = stub
    from experiment.batch_run import _expand_config_patterns
    return _expand_config_patterns


class TestBatchRunConfigPatterns(unittest.TestCase):
    def test_patterns_expand_and_exact_names_pass_through(self):
        _expand_config_patterns = _import_batch_run()
        from experiment.tracking import ALL_CONFIGS

        self.assertEqual(_expand_config_patterns(["*"]), ALL_CONFIGS)
        self.assertEqual(_expand_config_patterns(["no_such_config"]), ["no_such_config"])
        bio = _expand_config_patterns(["bio_*"])
        self.assertTrue(bio)
        self.assertTrue(all(n.startswith("bio_") for n in bio))

    def test_overlapping_patterns_do_not_duplicate(self):
        _expand_config_patterns = _import_batch_run()

        got = _expand_config_patterns(["bio_*", "bio_g*"])
        self.assertEqual(len(got), len(set(got)))

    def test_pattern_matching_nothing_is_an_error(self):
        _expand_config_patterns = _import_batch_run()

        with self.assertRaises(SystemExit):
            _expand_config_patterns(["definitely_no_match_*"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
