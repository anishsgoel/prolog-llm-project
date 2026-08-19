#!/usr/bin/env python
"""Build the London Underground abduction benchmark.

Pipeline
--------
    london_network.yaml            (ordered branch sequences, one readable table)
      -> canonical Prolog atoms    (station_atoms.csv)
      -> physical-adjacency edges  (consecutive pairs within a branch, deduped)
      -> express-skip assertions   (R2: never an edge that skips a station)
      -> tier subgraphs            (T1..T5, induced subgraphs of the full net)
      -> KB files + *.edges.json   (src/experiment/configs/kb/london_t*.pl)
      -> configs + validity gates  (src/experiment/configs/london_t*_*.yaml)
      -> benchmark_manifest.csv

Deterministic and re-runnable: given the same YAML it produces byte-identical
output.  All network data lives in ``london_network.yaml``; nothing about the
tube is hard-coded below except the canonicalisation rule.

Subcommands
-----------
    verify   data checks only (station counts, express skips, reference diff)
    edges    build the edge set and print graph statistics
    build    write tier KBs, configs and the manifest
    atoms    write station_atoms.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

_TOOLS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TOOLS_DIR.parent
_CONFIGS_DIR = _PROJECT_ROOT / "src" / "experiment" / "configs"
_KB_DIR = _CONFIGS_DIR / "kb"

NETWORK_YAML = _TOOLS_DIR / "london_network.yaml"
REFERENCE_EDGES = _TOOLS_DIR / "london_reference_edges.csv"

Edge = Tuple[str, str]  # always stored sorted, so it is an undirected key


# ---------------------------------------------------------------------------
# Canonicalisation to Prolog atoms
# ---------------------------------------------------------------------------

def _ascii_apostrophes(s: str) -> str:
    """Fold the Unicode apostrophe variants onto ASCII ``'``.

    Wikidata writes "Earl’s Court" with U+2019 while this project's YAML uses
    U+0027.  Without this the two spellings compare unequal and the reference
    diff reports five phantom Earl's Court discrepancies.
    """
    return s.replace("’", "'").replace("‘", "'").replace("ʼ", "'")


def canonical_atom(name: str) -> str:
    """Official station name -> Prolog atom.

    ``prolog.formula_parsing.parse_predicate`` only accepts ``[a-z_][a-zA-Z0-9_]*``
    functors, and ``prolog.prolog_utils.is_variable`` treats any argument
    starting with an uppercase letter *or an underscore* as a variable.  So the
    atom must start with a lowercase letter and contain only ``[a-z0-9_]``.

    Rule: lowercase; ``&`` -> ``and``; drop apostrophes and periods; every other
    non-alphanumeric run -> a single ``_``; collapse repeated ``_``; strip
    leading/trailing ``_``.
    """
    s = _ascii_apostrophes(name).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"['.]", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError(f"station name {name!r} canonicalises to the empty atom")
    if s[0].isdigit():
        raise ValueError(f"station name {name!r} canonicalises to {s!r}, which starts with a digit")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", s):
        raise ValueError(f"station name {name!r} canonicalises to the invalid atom {s!r}")
    return s


# ---------------------------------------------------------------------------
# Network model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExpressSkip:
    line: str
    frm: str
    to: str
    between: Tuple[str, ...]
    covered_by: str

    @property
    def pair(self) -> Edge:
        return tuple(sorted((self.frm, self.to)))  # type: ignore[return-value]


@dataclass
class Network:
    lines: Dict[str, dict]
    express_skips: List[ExpressSkip]
    known_parallel_structures: List[dict]
    zones: Dict[str, int] = field(default_factory=dict)
    tiers: Dict[str, dict] = field(default_factory=dict)

    def known_cycle_stations(self) -> List[Set[str]]:
        return [set(s["cycle"]) for s in self.known_parallel_structures]

    def line_names(self) -> List[str]:
        return [spec["name"] for spec in self.lines.values()]

    def stations_of_line(self, key: str) -> List[str]:
        seen: List[str] = []
        for branch in self.lines[key]["branches"]:
            for st in branch["stations"]:
                if st not in seen:
                    seen.append(st)
        return seen

    def all_stations(self) -> List[str]:
        seen: List[str] = []
        for key in self.lines:
            for st in self.stations_of_line(key):
                if st not in seen:
                    seen.append(st)
        return sorted(seen)


def load_network(path: Path = NETWORK_YAML) -> Network:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    skips = [
        ExpressSkip(
            line=s["line"], frm=s["from"], to=s["to"],
            between=tuple(s.get("between", [])), covered_by=s.get("covered_by", ""),
        )
        for s in data.get("express_skips", [])
    ]
    return Network(
        lines=data["lines"],
        express_skips=skips,
        known_parallel_structures=data.get("known_parallel_structures", []),
        zones=data.get("zones", {}),
        tiers=data.get("tiers", {}),
    )


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------

@dataclass
class Graph:
    """Undirected physical-adjacency graph over official station names."""

    edges: List[Edge] = field(default_factory=list)
    edge_lines: Dict[Edge, List[str]] = field(default_factory=dict)
    stations: List[str] = field(default_factory=list)

    def adjacency(self) -> Dict[str, Set[str]]:
        adj: Dict[str, Set[str]] = {s: set() for s in self.stations}
        for a, b in self.edges:
            adj[a].add(b)
            adj[b].add(a)
        return adj

    def degree(self) -> Dict[str, int]:
        return {s: len(n) for s, n in self.adjacency().items()}

    def induced(self, keep: Set[str]) -> "Graph":
        edges = [e for e in self.edges if e[0] in keep and e[1] in keep]
        return Graph(
            edges=edges,
            edge_lines={e: self.edge_lines[e] for e in edges},
            stations=sorted(s for s in self.stations if s in keep),
        )


def build_graph(net: Network) -> Tuple[Graph, List[Tuple[str, Edge]]]:
    """Derive the undirected edge set from the ordered branch sequences.

    Returns ``(graph, suppressed)`` where ``suppressed`` lists the
    ``(line, pair)`` consecutive pairs dropped because they are declared express
    skips.  Every edge is a pair of *consecutive* stations within one branch, so
    R2 (no shortcut edges) holds by construction, modulo the express skips this
    function removes.
    """
    skip_index: Dict[str, Set[Edge]] = defaultdict(set)
    for s in net.express_skips:
        skip_index[s.line].add(s.pair)

    edge_lines: Dict[Edge, List[str]] = {}
    suppressed: List[Tuple[str, Edge]] = []

    for key, spec in net.lines.items():
        line_name = spec["name"]
        for branch in spec["branches"]:
            seq = branch["stations"]
            for u, v in zip(seq, seq[1:]):
                if u == v:
                    raise ValueError(f"{line_name}/{branch['name']}: station {u!r} repeated back-to-back")
                pair: Edge = tuple(sorted((u, v)))  # type: ignore[assignment]
                if pair in skip_index[line_name]:
                    suppressed.append((line_name, pair))
                    continue
                edge_lines.setdefault(pair, [])
                if line_name not in edge_lines[pair]:
                    edge_lines[pair].append(line_name)

    stations = net.all_stations()
    graph = Graph(edges=sorted(edge_lines), edge_lines=edge_lines, stations=stations)
    return graph, suppressed


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_express_skips(net: Network, graph: Graph) -> List[str]:
    """R2: no declared express skip may appear as an edge, and each skipped
    corridor must actually be present as a chain through the `between` stations."""
    problems: List[str] = []
    present = set(graph.edges)
    adj = graph.adjacency()
    for s in net.express_skips:
        if s.pair in present:
            problems.append(
                f"R2 VIOLATION: express skip {s.frm} -- {s.to} ({s.line}) is in the edge set"
            )
        chain = [s.frm, *s.between, s.to]
        for u, v in zip(chain, chain[1:]):
            if u not in adj or v not in adj:
                problems.append(f"express skip {s.frm}->{s.to}: station {u!r}/{v!r} missing from graph")
            elif v not in adj[u]:
                problems.append(
                    f"express skip {s.frm}->{s.to}: corridor broken, no edge {u} -- {v} "
                    f"(should be supplied by the {s.covered_by})"
                )
    return problems


def find_parallel_edges(graph: Graph, max_len: int = 3) -> List[dict]:
    """Generic shortcut detector.

    Flags any edge (u, w) for which the graph *minus that edge* still contains a
    u-w path of length 2..max_len whose intermediate stations all lie on a line
    that serves both u and w.  Reported for manual review, never auto-deleted:
    the tube genuinely has parallel tunnels (Northern line Bank vs Charing Cross
    branch, the Heathrow T4 loop).
    """
    adj = graph.adjacency()
    station_lines: Dict[str, Set[str]] = defaultdict(set)
    for (a, b), lines in graph.edge_lines.items():
        station_lines[a].update(lines)
        station_lines[b].update(lines)

    hits: List[dict] = []
    for u, w in graph.edges:
        shared = station_lines[u] & station_lines[w]
        if not shared:
            continue
        # BFS for alternative paths of length 2..max_len avoiding the direct edge
        found: List[List[str]] = []
        stack: List[List[str]] = [[u]]
        while stack:
            path = stack.pop()
            if len(path) > max_len:
                continue
            last = path[-1]
            for nxt in sorted(adj[last]):
                if nxt in path:
                    continue
                if last == u and nxt == w:
                    continue  # the direct edge itself
                if nxt == w:
                    if len(path) >= 2:
                        found.append(path[1:])
                    continue
                if len(path) < max_len:
                    stack.append(path + [nxt])
        for mids in found:
            if not mids:
                continue
            if all(shared & station_lines[m] for m in mids):
                hits.append({"edge": [u, w], "via": mids,
                             "lines": sorted(shared & set.intersection(*(station_lines[m] for m in mids)))})
    # dedupe by (edge, via)
    seen = set()
    out = []
    for h in hits:
        k = (tuple(h["edge"]), tuple(h["via"]))
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
    return out


def check_line_station_counts(net: Network, expected: Dict[str, int]) -> List[str]:
    problems = []
    for key, spec in net.lines.items():
        got = len(net.stations_of_line(key))
        want = expected.get(spec["name"])
        if want is not None and got != want:
            problems.append(f"{spec['name']}: {got} stations in YAML, {want} expected")
    return problems


def _norm_reference_name(label: str) -> str:
    """Normalise an external reference label to this benchmark's station name."""
    s = _ascii_apostrophes(label).strip()
    s = re.sub(r"\s+(tube|railway|underground|metro)\s+station$", "", s, flags=re.I)
    s = re.sub(r"\s+station$", "", s, flags=re.I)
    s = re.sub(r"^St\.\s+", "St ", s)
    s = s.replace(" and Whetstone", " & Whetstone")
    fixes = {
        "Paddington (Bishop's Road)": "Paddington",
        "Paddington (Praed Street)": "Paddington",
        "Queen's Park (England)": "Queen's Park",
        "Kew Gardens (London)": "Kew Gardens",
        "Richmond (London)": "Richmond",
        "Wimbledon (London)": "Wimbledon",
        "South Acton (England)": "South Acton",
    }
    return fixes.get(s, s)


def diff_against_reference(graph: Graph, path: Path = REFERENCE_EDGES) -> Optional[dict]:
    """Diff the derived edge set against the checked-in reference adjacency."""
    if not path.exists():
        return None
    ref: Set[Edge] = set()
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(l for l in fh if not l.startswith("#")):
            a = _norm_reference_name(row["a"])
            b = _norm_reference_name(row["b"])
            if a == b:
                continue
            ref.add(tuple(sorted((a, b))))  # type: ignore[arg-type]
    ours = set(graph.edges)
    return {
        "reference_edges": len(ref),
        "our_edges": len(ours),
        "missing_from_ours": sorted(ref - ours),
        "extra_in_ours": sorted(ours - ref),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _connected_components(graph: Graph) -> List[Set[str]]:
    adj = graph.adjacency()
    seen: Set[str] = set()
    comps: List[Set[str]] = []
    for s in graph.stations:
        if s in seen:
            continue
        comp: Set[str] = set()
        dq = deque([s])
        seen.add(s)
        while dq:
            cur = dq.popleft()
            comp.add(cur)
            for n in adj[cur]:
                if n not in seen:
                    seen.add(n)
                    dq.append(n)
        comps.append(comp)
    return comps


def report_graph(net: Network, graph: Graph, suppressed: List[Tuple[str, Edge]]) -> None:
    deg = graph.degree()
    dist = Counter(deg.values())
    comps = _connected_components(graph)

    print("=" * 78)
    print("EDGE SET")
    print("=" * 78)
    print(f"  stations (nodes)        : {len(graph.stations)}")
    print(f"  physical edges (undir.) : {len(graph.edges)}")
    print(f"  prolog facts (2 x edges): {2 * len(graph.edges)}")
    print(f"  connected components    : {len(comps)} "
          f"(sizes: {sorted((len(c) for c in comps), reverse=True)})")
    print(f"  suppressed express hops : {len(suppressed)}")
    for line, pair in suppressed:
        print(f"      - {line}: {pair[0]} -- {pair[1]}")

    print("\n  degree distribution:")
    for d in sorted(dist):
        print(f"      degree {d}: {dist[d]:3d} stations")

    deg1 = sorted(s for s, d in deg.items() if d == 1)
    print(f"\n  degree-1 stations ({len(deg1)}) - excluded from edge-omission candidates (R3):")
    for s in deg1:
        print(f"      - {s}")

    top = sorted(deg.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    print("\n  highest-degree stations (junctions):")
    for s, d in top:
        print(f"      {d}  {s}")


def report_verification(net: Network, graph: Graph, suppressed: List[Tuple[str, Edge]],
                        expected_counts: Dict[str, int]) -> int:
    rc = 0
    print("=" * 78)
    print("DATA VERIFICATION")
    print("=" * 78)

    print("\n[1] Station counts per line (YAML vs reference)")
    for key, spec in net.lines.items():
        got = len(net.stations_of_line(key))
        want = expected_counts.get(spec["name"])
        flag = "ok " if want is None or got == want else "ERR"
        if flag == "ERR":
            rc = 1
        print(f"    {flag} {spec['name']:20s} {got:3d}" + (f"  (reference {want})" if want else ""))
    print(f"    -- distinct stations in network: {len(graph.stations)}")

    print("\n[2] Canonicalisation (all station names -> valid Prolog atoms)")
    atoms: Dict[str, str] = {}
    clashes = []
    for st in graph.stations:
        a = canonical_atom(st)
        if a in atoms and atoms[a] != st:
            clashes.append((atoms[a], st, a))
        atoms[a] = st
    if clashes:
        rc = 1
        for x, y, a in clashes:
            print(f"    ERR atom collision: {x!r} and {y!r} both -> {a}")
    else:
        print(f"    ok  {len(atoms)} distinct atoms, no collisions")

    print("\n[3] Express skips (R2)")
    problems = check_express_skips(net, graph)
    if problems:
        rc = 1
        for p in problems:
            print(f"    ERR {p}")
    else:
        for s in net.express_skips:
            print(f"    ok  {s.line:14s} {s.frm} -/-> {s.to}  "
                  f"(via {', '.join(s.between)} on the {s.covered_by})")

    print("\n[4] Generic parallel-edge detector (manual review, not auto-deleted)")
    cycles = net.known_cycle_stations()
    hits = find_parallel_edges(graph)
    unexplained = []
    for h in hits:
        involved = set(h["edge"]) | set(h["via"])
        known = any(involved <= cyc for cyc in cycles)
        tag = "known" if known else "NEW  "
        if not known:
            unexplained.append(h)
        print(f"    {tag} {h['edge'][0]} -- {h['edge'][1]}  (alt. path via {', '.join(h['via'])})")
    if not hits:
        print("    (none)")
    if unexplained:
        print(f"    !! {len(unexplained)} hit(s) not covered by a declared "
              f"known_parallel_structure - review before shipping")

    print("\n[5] Closed / unbuilt stations must be absent")
    forbidden = [
        "Aldwych", "Brompton Road", "Down Street", "York Road", "British Museum",
        "Ongar", "North Weald", "Blake Hall", "Cassiobridge", "Watford Vicarage Road",
        "Lewisham", "Wood Lane (Central line)", "Swiss Cottage (1868-1940)",
        "Marlborough Road", "Lord's", "City Road", "South Kentish Town",
    ]
    present = set(graph.stations)
    bad = [f for f in forbidden if f in present]
    if bad:
        rc = 1
        for f in bad:
            print(f"    ERR closed/unbuilt station present: {f}")
    else:
        print(f"    ok  none of the {len(forbidden)} checked closed/unbuilt stations present")

    print("\n[6] Naming collisions that must stay distinct")
    pairs = [("Shepherd's Bush", "Shepherd's Bush Market"),
             ("Edgware Road", "Edgware Road (Bakerloo)"),
             ("Edgware", "Edgware Road")]
    for a, b in pairs:
        ok = a in present and b in present and canonical_atom(a) != canonical_atom(b)
        if not ok:
            rc = 1
        print(f"    {'ok ' if ok else 'ERR'} {a!r} vs {b!r} -> "
              f"{canonical_atom(a)} / {canonical_atom(b)}")

    print("\n[7] Diff against checked-in reference adjacency")
    diff = diff_against_reference(graph)
    if diff is None:
        print(f"    -- no reference file at {REFERENCE_EDGES.name}, skipped")
    else:
        print(f"    reference edges: {diff['reference_edges']}, ours: {diff['our_edges']}")
        print(f"    in reference but not ours ({len(diff['missing_from_ours'])}):")
        for a, b in diff["missing_from_ours"]:
            print(f"        - {a} -- {b}")
        print(f"    in ours but not reference ({len(diff['extra_in_ours'])}):")
        for a, b in diff["extra_in_ours"]:
            print(f"        + {a} -- {b}")

    print("\n[8] Connectivity")
    comps = _connected_components(graph)
    if len(comps) != 1:
        rc = 1
        print(f"    ERR graph has {len(comps)} components")
        for c in sorted(comps, key=len)[:-1]:
            print(f"        stranded: {sorted(c)}")
    else:
        print("    ok  the whole network is one connected component")

    return rc


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------

@dataclass
class Tier:
    key: str
    file: str
    title: str
    graph: Graph
    stranded: List[str]


def build_tier(net: Network, full: Graph, key: str, spec: dict,
               zones: Dict[str, int]) -> Tier:
    """Cut one tier as an INDUCED SUBGRAPH of the full network.

    Induced (not re-derived) is what makes tiers comparable: an edge is in the
    tier iff both its endpoints are, so an edge means the same thing in T1 and
    in T5.  Anything the cut strands from the main component is dropped and
    reported - a disconnected KB would make some goals trivially unprovable for
    reasons that have nothing to do with the omitted facts.
    """
    keep: Set[str] = set()
    for line_key in spec.get("lines", []):
        keep.update(net.stations_of_line(line_key))
    for ref in spec.get("branches", []):
        line_key, _, branch_name = ref.partition("/")
        branches = net.lines[line_key]["branches"]
        match = next((b for b in branches if b["name"] == branch_name), None)
        if match is None:
            raise KeyError(f"tier {key}: no branch {ref!r}")
        keep.update(match["stations"])

    max_zone = spec.get("max_zone")
    if max_zone is not None:
        keep = {s for s in keep if zones.get(s, 99) <= max_zone}

    sub = full.induced(keep)
    comps = _connected_components(sub)
    stranded: List[str] = []
    if len(comps) > 1:
        main = max(comps, key=len)
        stranded = sorted(s for c in comps if c is not main for s in c)
        sub = sub.induced(main)

    return Tier(key=key, file=spec["file"], title=spec["title"], graph=sub, stranded=stranded)


# ---------------------------------------------------------------------------
# KB emission
# ---------------------------------------------------------------------------

_CONNECTED_COMMENT = (
    "connected/2 = ADJACENT STOPS ONLY. Two stations that are immediate physical "
    "neighbours on a London Underground line, with no other station between them. "
    "NEVER a shortcut edge that skips an intermediate station. Every edge is listed "
    "in both directions."
)
_REACHABLE_BASE_COMMENT = (
    "reachable/2 = PATH EXISTS. One-hop reachable comes only from connected/2."
)
# program_parsing keys predicate comments by functor/arity and the LAST clause
# carrying a comment wins, so this one is what the prompts actually show for
# reachable/2. It therefore restates the base case as well.
_REACHABLE_REC_COMMENT = (
    "reachable/2 is the transitive closure of connected/2: a multi-hop path that "
    "follows only adjacent edges. One hop comes from connected/2 directly."
)


def emit_kb(tier: Tier) -> Tuple[str, List[dict]]:
    """Render a tier as KB text, plus the edge -> (fact_ab, fact_ba) index.

    The two directions of each edge are emitted as ADJACENT, consecutively
    numbered facts (2k-1, 2k) so an omission set follows from an edge index by
    arithmetic.  The two reachable/2 rules come last, after every fact, and
    there is deliberately NO symmetry rule (R5): `connected(X, Y) :- connected(Y, X).`
    makes proving connected(a, b) recurse through connected(b, a) and back,
    bounded only by the depth limit.
    """
    lines: List[str] = []
    index: List[dict] = []
    num = 1
    for i, (a_name, b_name) in enumerate(tier.graph.edges):
        a, b = canonical_atom(a_name), canonical_atom(b_name)
        comment = f"  # {_CONNECTED_COMMENT}" if num == 1 else ""
        lines.append(f"{num}. connected({a}, {b}).{comment}")
        lines.append(f"{num + 1}. connected({b}, {a}).")
        index.append({
            "index": i,
            "a_name": a_name, "b_name": b_name,
            "a": a, "b": b,
            "fact_ab": num, "fact_ba": num + 1,
            "lines": tier.graph.edge_lines[(a_name, b_name)],
        })
        num += 2

    lines.append(f"{num}. reachable(X, Y) :- connected(X, Y).  # {_REACHABLE_BASE_COMMENT}")
    lines.append(f"{num + 1}. reachable(X, Z) :- connected(X, Y), reachable(Y, Z).  "
                 f"# {_REACHABLE_REC_COMMENT}")
    return "\n".join(lines) + "\n", index


def write_tier_kb(tier: Tier, kb_dir: Path) -> List[dict]:
    text, index = emit_kb(tier)
    kb_path = kb_dir / tier.file
    kb_path.write_text(text, encoding="utf-8")

    sidecar = kb_path.with_suffix(".edges.json")
    sidecar.write_text(json.dumps({
        "kb": tier.file,
        "tier": tier.key,
        "title": tier.title,
        "n_stations": len(tier.graph.stations),
        "n_edges": len(tier.graph.edges),
        "n_facts": 2 * len(tier.graph.edges),
        "rule_fact_ids": [2 * len(tier.graph.edges) + 1, 2 * len(tier.graph.edges) + 2],
        "stranded_by_cut": tier.stranded,
        "edges": index,
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return index


# ---------------------------------------------------------------------------
# station_atoms.csv
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

DEPTH_BANDS = {"short": (3, 5), "medium": (6, 10), "long": (11, 20)}


def required_proof_depth(hops: int) -> int:
    """Minimum Solver max_depth at which an `hops`-hop reachable/2 proof is found.

    Measured on the tier KBs: hops 1..5 need depth 1, 3, 4, 6, 7 - i.e.
    ``hops + hops // 2``.  Each intermediate hop costs one rule-2 application
    plus one fact unification, while the final hop closes through rule 1.

    This matters more than it looks: `Solver` reports a goal as UNPROVABLE when
    it merely ran out of depth, so a max_depth that is too small silently fails
    gate 1 and drops a perfectly good config.
    """
    return hops + hops // 2


def _bfs_path(adj: Dict[str, Set[str]], src: str, dst: str) -> Optional[List[str]]:
    if src == dst:
        return [src]
    prev = {src: None}
    dq = deque([src])
    while dq:
        cur = dq.popleft()
        for nxt in sorted(adj[cur]):
            if nxt in prev:
                continue
            prev[nxt] = cur
            if nxt == dst:
                path = [dst]
                while path[-1] is not None and prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            dq.append(nxt)
    return None


def _connected_after(graph: Graph, removed: Set[Edge], src: str, dst: str) -> bool:
    """Is dst still reachable from src once `removed` is deleted?

    This is exactly gate 2.  Because every physical edge is emitted as BOTH
    connected(a,b) and connected(b,a), and an omission always drops both facts,
    directed Prolog reachability coincides with undirected connectivity - so
    this cheap check is equivalent to running the solver, and is used to filter
    candidates before spending a subprocess on the real thing.
    """
    adj = graph.adjacency()
    seen = {src}
    dq = deque([src])
    while dq:
        cur = dq.popleft()
        for nxt in adj[cur]:
            if tuple(sorted((cur, nxt))) in removed or nxt in seen:
                continue
            if nxt == dst:
                return True
            seen.add(nxt)
            dq.append(nxt)
    return False


def _path_edges(path: List[str]) -> List[Edge]:
    return [tuple(sorted((u, v))) for u, v in zip(path, path[1:])]  # type: ignore[misc]


def _omission_keeps_every_station(graph: Graph, removed: Set[Edge]) -> Tuple[bool, List[str]]:
    """R3: every station must still appear at least once in the ablated KB.

    A station whose last edge is omitted vanishes from the program entirely, and
    with allow_new_constants=false the model is then forbidden from naming it -
    the task becomes unsolvable for a reason that has nothing to do with
    abduction.
    """
    deg = Counter()
    for e in graph.edges:
        if e in removed:
            continue
        deg[e[0]] += 1
        deg[e[1]] += 1
    orphans = sorted(s for s in graph.stations if deg[s] == 0)
    return (not orphans), orphans


def _candidate_omissions(graph: Graph, src: str, dst: str, k: int,
                         placement: str, limit: int = 6) -> List[List[Edge]]:
    """Omission sets of exactly k edges that break every src->dst route.

    placement="onpath"  - k NON-ADJACENT edges of the shortest path, so repairing
                          along the original corridor requires a chain of k
                          separate hypotheses.  They must be non-adjacent
                          because two consecutive omitted edges would strip the
                          station between them of its last remaining edge, which
                          gate 3 (R3) rejects - on a corridor like the Bakerloo
                          that rules out every consecutive window.
    placement="spread"  - edges taken from two or more distinct routes, so no
                          single corridor repair suffices.  Built from a minimum
                          cut (found by iteratively removing shortest paths),
                          topped up from the shortest path if the cut is smaller
                          than k.
    """
    adj = graph.adjacency()
    path = _bfs_path(adj, src, dst)
    if path is None or len(path) < 2:
        return []
    p_edges = _path_edges(path)
    out: List[List[Edge]] = []

    if placement == "onpath":
        from itertools import combinations
        # k edges of the path, pairwise non-adjacent (indices at least 2 apart),
        # so no intermediate station loses every edge it has.
        for idx in combinations(range(len(p_edges)), k):
            if any(b - a < 2 for a, b in zip(idx, idx[1:])):
                continue
            pick = [p_edges[i] for i in idx]
            if not _connected_after(graph, set(pick), src, dst):
                out.append(pick)
            if len(out) >= limit:
                break
        return out

    # spread: peel edge-disjoint routes to get a cut that straddles them
    work = Graph(edges=list(graph.edges), edge_lines=dict(graph.edge_lines),
                 stations=list(graph.stations))
    cut: List[Edge] = []
    removed: Set[Edge] = set()
    for _ in range(k):
        cur = _bfs_path(work.adjacency(), src, dst)
        if cur is None:
            break
        cur_edges = [e for e in _path_edges(cur) if e not in removed]
        if not cur_edges:
            break
        # take the middle edge of the route, keeping the cut away from termini
        pick = cur_edges[len(cur_edges) // 2]
        cut.append(pick)
        removed.add(pick)
        work = Graph(edges=[e for e in work.edges if e != pick],
                     edge_lines={e: v for e, v in work.edge_lines.items() if e != pick},
                     stations=list(work.stations))
        if not _connected_after(graph, removed, src, dst):
            break

    if len(cut) < k:
        for e in p_edges:
            if len(cut) >= k:
                break
            if e not in cut:
                cut.append(e)
    cut = cut[:k]
    if len(cut) == k and not _connected_after(graph, set(cut), src, dst):
        out.append(sorted(cut))
    return out


def _classify_goal(graph: Graph, zones: Dict[str, int], path: List[str]) -> dict:
    deg = graph.degree()
    hops = len(path) - 1
    band = next((b for b, (lo, hi) in DEPTH_BANDS.items() if lo <= hops <= hi), "very_long")
    junctions = [s for s in path[1:-1] if deg[s] >= 3]
    src, dst = path[0], path[-1]
    shortcut_plausible = (
        zones.get(src, 99) == 1 and zones.get(dst, 99) == 1
        and deg[src] >= 3 and deg[dst] >= 3
    )
    return {
        "hops": hops,
        "depth_band": band,
        "crosses_junction": bool(junctions),
        "junctions": junctions,
        "shortcut_plausible": shortcut_plausible,
    }


def _goal_candidates(tier: Tier, zones: Dict[str, int]) -> List[dict]:
    """All station pairs of the tier, classified, in a deterministic order."""
    adj = tier.graph.adjacency()
    stations = tier.graph.stations
    out: List[dict] = []
    for i, src in enumerate(stations):
        # single BFS from src gives every shortest path length at once
        dist = {src: 0}
        dq = deque([src])
        while dq:
            cur = dq.popleft()
            for nxt in sorted(adj[cur]):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    dq.append(nxt)
        for dst in stations[i + 1:]:
            hops = dist.get(dst)
            if hops is None or hops < 3 or hops > 20:
                continue
            path = _bfs_path(adj, src, dst)
            if path is None:
                continue
            info = _classify_goal(tier.graph, zones, path)
            info.update({"src": src, "dst": dst, "path": path})
            out.append(info)
    return out


def _select_goals(candidates: List[dict], per_cell: int) -> List[dict]:
    """Stratified, deterministic pick across depth band x junction x shortcut."""
    cells: Dict[tuple, List[dict]] = defaultdict(list)
    for c in candidates:
        cells[(c["depth_band"], c["crosses_junction"], c["shortcut_plausible"])].append(c)
    chosen: List[dict] = []
    for key in sorted(cells, key=lambda k: (str(k[0]), k[1], k[2])):
        bucket = sorted(cells[key], key=lambda c: (c["hops"], c["src"], c["dst"]))
        if not bucket:
            continue
        step = max(1, len(bucket) // per_cell)
        picked = bucket[::step][:per_cell]
        chosen.extend(picked)
    return chosen


def write_station_atoms(graph: Graph, net: Network, out: Path) -> None:
    station_lines: Dict[str, Set[str]] = defaultdict(set)
    for key, spec in net.lines.items():
        for st in net.stations_of_line(key):
            station_lines[st].add(spec["name"])
    deg = graph.degree()
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["official_name", "prolog_atom", "lines", "degree"])
        for st in graph.stations:
            w.writerow([st, canonical_atom(st), "; ".join(sorted(station_lines[st])), deg[st]])


# ---------------------------------------------------------------------------
# Validity gates
# ---------------------------------------------------------------------------

def _import_project():
    """Put the repo on sys.path and return the pieces the gates need."""
    for p in (str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import config as project_config
    from experiment.groundtruth import solve_ground_truth
    project_config.init_config()
    return solve_ground_truth


@dataclass
class GateResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    gt_depth: Optional[int] = None
    gt_time_s: Optional[float] = None
    gt_proof: str = ""


def run_gates(tier: Tier, kb_text: str, goal: dict, omitted: List[Edge],
              edge_index: Dict[Edge, dict], gt_cache: Dict[str, GateResult],
              solve_ground_truth, gt_timeout: float, max_depth: int) -> GateResult:
    """The five validity gates from the task spec.

    Gates 2-4 are cheap graph properties; gates 1 and 5 need a real proof
    search, so their result is cached per (tier, goal) - the ground truth runs
    on the COMPLETE tier KB and does not depend on which facts are omitted.
    """
    import time

    reasons: List[str] = []
    src, dst = goal["src"], goal["dst"]
    goal_str = f"reachable({canonical_atom(src)}, {canonical_atom(dst)})"
    removed = set(omitted)

    # Gate 2: the goal must NOT be provable on the ablated KB.
    if _connected_after(tier.graph, removed, src, dst):
        reasons.append("gate2: goal still provable without the omitted facts")

    # Gate 3: every station still appears in the ablated KB.
    ok3, orphans = _omission_keeps_every_station(tier.graph, removed)
    if not ok3:
        reasons.append(f"gate3: stations orphaned by the omission: {', '.join(orphans)}")

    # Gate 4: no omitted edge was the last edge of a degree-1 station.
    deg = tier.graph.degree()
    for a, b in omitted:
        for s in (a, b):
            if deg[s] == 1:
                reasons.append(f"gate4: {s} has degree 1; omitting its only edge removes it")

    # Gates 1 and 5: ground truth on the complete tier KB, inside the budget.
    # Skipped when a cheap gate already rejected this candidate and the goal is
    # not cached - the ground-truth search is by far the most expensive step
    # here, and its verdict would not change the outcome.
    if reasons and goal_str not in gt_cache:
        return GateResult(passed=False, reasons=reasons)
    if goal_str not in gt_cache:
        t0 = time.perf_counter()
        res = solve_ground_truth(kb_text, goal_str, max_depth=max_depth, timeout_s=gt_timeout)
        elapsed = time.perf_counter() - t0
        if res is None:
            # solve_ground_truth returns None both for a real timeout and for a
            # worker that died (e.g. a missing dependency in the environment).
            # Returning far under the budget means it died - reporting that as a
            # tractability result would quietly blame the solver for a broken
            # install, so it is raised instead.
            if elapsed < 0.5 * gt_timeout:
                raise RuntimeError(
                    f"ground-truth worker died after {elapsed:.1f}s (budget {gt_timeout:g}s) "
                    f"for {goal_str} on {tier.file}. This is an environment failure, not a "
                    f"timeout - run the goal directly to see the child's traceback."
                )
            gt_cache[goal_str] = GateResult(
                False, [f"gate5: ground truth exceeded {gt_timeout:g}s"], None, elapsed, "")
        else:
            success, depth, proof_str, _leaves = res
            gt_cache[goal_str] = GateResult(
                bool(success),
                [] if success else ["gate1: goal not provable on the complete tier KB"],
                depth, elapsed, proof_str,
            )
    gt = gt_cache[goal_str]
    reasons.extend(gt.reasons)

    return GateResult(passed=not reasons, reasons=reasons,
                      gt_depth=gt.gt_depth, gt_time_s=gt.gt_time_s, gt_proof=gt.gt_proof)


# ---------------------------------------------------------------------------
# Config emission
# ---------------------------------------------------------------------------

def _config_text(name: str, goal: dict, tier: Tier, omitted: List[Edge],
                 edge_index: Dict[Edge, dict], fact_ids: List[int],
                 info: dict, max_depth: Optional[int]) -> str:
    edge_desc = "\n".join(
        f"#   {edge_index[e]['a_name']} -- {edge_index[e]['b_name']} "
        f"(facts {edge_index[e]['fact_ab']}, {edge_index[e]['fact_ba']})"
        for e in omitted
    )
    header = (
        f"# {tier.key.upper()} - {tier.title}\n"
        f"#\n"
        f"# Goal: {goal['src']} -> {goal['dst']}  "
        f"({info['hops']} hops, {info['depth_band']} path)\n"
        f"# Shortest route: {' -> '.join(goal['path'])}\n"
        f"# Omitted edges ({len(omitted)}), both directions each:\n"
        f"{edge_desc}\n"
        f"# Placement: {info['placement']}  |  crosses junction: "
        f"{'yes (' + ', '.join(info['junctions']) + ')' if info['crosses_junction'] else 'no'}\n"
        f"# Plausible direct-shortcut temptation: "
        f"{'yes - both endpoints are Zone 1 interchanges' if info['shortcut_plausible'] else 'no'}\n"
        f"#\n"
        f"# allow_new_constants is false: every station the model may need is still\n"
        f"# named somewhere in the ablated program (gate 3).\n"
    )
    body = [
        f"name: {name}",
        f'goal: "reachable({canonical_atom(goal["src"])}, {canonical_atom(goal["dst"])})"',
        f"kb_file: kb/{tier.file}",
        "propose_facts: connected/2",
        f"omit_fact_ids: {fact_ids}",
        "allow_new_constants: false",
    ]
    if max_depth is not None:
        body.append("solver:")
        body.append(f"  max_depth_ceiling: {max_depth}")
    return header + "\n".join(body) + "\n"


def generate_configs(tier: Tier, kb_text: str, edge_index: Dict[Edge, dict],
                     zones: Dict[str, int], solve_ground_truth, gt_timeout: float,
                     target: int, per_cell: int) -> Tuple[List[dict], List[dict]]:
    """Emit gate-passing configs for one tier; return (emitted, rejected)."""
    # Re-runnability: config names carry a per-tier sequence number, so a run
    # that emits fewer configs than the last one would otherwise leave stale
    # YAML behind and the directory would no longer match the manifest.
    for stale in _CONFIGS_DIR.glob(f"london_{tier.key}_*.yaml"):
        stale.unlink()

    candidates = _select_goals(_goal_candidates(tier, zones), per_cell)
    # Shortest goals first: ground-truth cost grows steeply with hop count, so
    # this both front-loads the cheap work and lets the tractability wall below
    # cut off everything longer as soon as one goal blows the budget.
    candidates.sort(key=lambda c: (c["hops"], c["src"], c["dst"]))
    gt_cache: Dict[str, GateResult] = {}
    emitted: List[dict] = []
    rejected: List[dict] = []
    seq = 0
    wall_hops: Optional[int] = None

    plan = [(k, placement) for k in (1, 2, 3, 5) for placement in ("onpath", "spread")]

    for goal in candidates:
        if len(emitted) >= target:
            break
        if wall_hops is not None and goal["hops"] >= wall_hops:
            rejected.append({
                "tier": tier.key, "kb": tier.file, "config_name": "",
                "goal_src": goal["src"], "goal_dst": goal["dst"],
                "shortest_path_hops": goal["hops"], "depth_band": goal["depth_band"],
                "gates_passed": "",
                "gate_failures": f"gate5: skipped, tier hit the ground-truth "
                                 f"tractability wall at {wall_hops} hops",
            })
            continue
        # Tight but sufficient for the ground-truth call; the config itself gets
        # extra headroom because a repair through hypothesised facts may be
        # longer than the original shortest path.
        gt_depth_budget = required_proof_depth(goal["hops"]) + 2
        max_depth = required_proof_depth(goal["hops"]) + 6
        for k, placement in plan:
            if len(emitted) >= target:
                break
            for omitted in _candidate_omissions(tier.graph, goal["src"], goal["dst"],
                                                k, placement, limit=1):
                gate = run_gates(tier, kb_text, goal, omitted, edge_index, gt_cache,
                                 solve_ground_truth, gt_timeout, gt_depth_budget)
                if any(r.startswith("gate5") for r in gate.reasons):
                    wall_hops = goal["hops"]
                info = dict(goal)
                info["placement"] = placement
                fact_ids = sorted(
                    fid for e in omitted
                    for fid in (edge_index[e]["fact_ab"], edge_index[e]["fact_ba"])
                )
                record = {
                    "tier": tier.key, "kb": tier.file,
                    "goal_src": goal["src"], "goal_dst": goal["dst"],
                    "goal": f"reachable({canonical_atom(goal['src'])}, "
                            f"{canonical_atom(goal['dst'])})",
                    "n_omitted_edges": k, "placement": placement,
                    "omitted_fact_ids": fact_ids,
                    "omitted_edges": "; ".join(
                        f"{edge_index[e]['a_name']}--{edge_index[e]['b_name']}" for e in omitted),
                    "shortest_path_hops": goal["hops"],
                    "depth_band": goal["depth_band"],
                    "crosses_junction": goal["crosses_junction"],
                    "shortcut_plausible": goal["shortcut_plausible"],
                    "gt_proof_depth": gate.gt_depth,
                    "gt_solve_time_s": None if gate.gt_time_s is None else round(gate.gt_time_s, 2),
                    "gates_passed": "all" if gate.passed else "",
                    "gate_failures": "; ".join(gate.reasons),
                    "max_depth_ceiling": max_depth,
                }
                if not gate.passed:
                    rejected.append(record)
                    continue
                seq += 1
                name = f"london_{tier.key}_{k}edge_{placement}_{seq:02d}"
                record["config_name"] = name
                record["config_file"] = f"{name}.yaml"
                text = _config_text(name, goal, tier, omitted, edge_index,
                                    fact_ids, info, max_depth)
                (_CONFIGS_DIR / f"{name}.yaml").write_text(text, encoding="utf-8")
                emitted.append(record)
    return emitted, rejected


MANIFEST_FIELDS = [
    "config_name", "config_file", "tier", "kb", "goal", "goal_src", "goal_dst",
    "n_omitted_edges", "placement", "omitted_fact_ids", "omitted_edges",
    "shortest_path_hops", "depth_band", "crosses_junction", "shortcut_plausible",
    "gt_proof_depth", "gt_solve_time_s", "max_depth_ceiling",
    "gates_passed", "gate_failures",
]


def write_manifest(rows: List[dict], out: Path) -> None:
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Station counts per line, from the Wikipedia "List of London Underground
# stations" table (which lists 272 rows).  Adjusted for the two identity merges
# documented in london_network.yaml: Hammersmith and Paddington are each one
# station here but two rows there, so Circle/H&C/District/Piccadilly counts are
# taken over merged names.
REFERENCE_LINE_COUNTS = {
    "Bakerloo": 25,
    "Central": 49,
    "Circle": 35,            # 36 rows, minus the duplicate Paddington
    "District": 60,
    "Hammersmith & City": 29,
    "Jubilee": 27,
    "Metropolitan": 34,
    "Northern": 52,
    "Piccadilly": 53,
    "Victoria": 16,
    "Waterloo & City": 2,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["verify", "edges", "atoms", "tiers", "build"],
                    nargs="?", default="verify")
    ap.add_argument("--network", type=Path, default=NETWORK_YAML)
    ap.add_argument("--tiers", nargs="+", default=None, metavar="KEY",
                    help="Restrict `tiers`/`build` to these tier keys (default: all).")
    ap.add_argument("--gt-timeout", type=float, default=300.0, metavar="SECONDS",
                    help="Ground-truth budget per goal (default 300, matching "
                         "experiment.groundtruth.GROUND_TRUTH_TIMEOUT_S).")
    ap.add_argument("--configs-per-tier", type=int, default=14, metavar="N")
    ap.add_argument("--goals-per-cell", type=int, default=3, metavar="N",
                    help="Goals sampled per (depth band x junction x shortcut) cell.")
    args = ap.parse_args(argv)

    net = load_network(args.network)
    graph, suppressed = build_graph(net)

    if args.command == "verify":
        return report_verification(net, graph, suppressed, REFERENCE_LINE_COUNTS)
    if args.command == "edges":
        report_graph(net, graph, suppressed)
        return 0
    if args.command == "atoms":
        out = _PROJECT_ROOT / "station_atoms.csv"
        write_station_atoms(graph, net, out)
        print(f"wrote {out} ({len(graph.stations)} stations)")
        return 0

    tier_keys = args.tiers or list(net.tiers)
    tiers = [build_tier(net, graph, k, net.tiers[k], net.zones) for k in tier_keys]

    if args.command == "tiers":
        print(f"{'tier':6s} {'stations':>8s} {'edges':>6s} {'facts':>6s}  title")
        for t in tiers:
            print(f"{t.key:6s} {len(t.graph.stations):8d} {len(t.graph.edges):6d} "
                  f"{2 * len(t.graph.edges):6d}  {t.title}")
            if t.stranded:
                print(f"       stranded by the cut ({len(t.stranded)}): {', '.join(t.stranded)}")
        return 0

    # build
    solve_ground_truth = _import_project()
    _KB_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []
    all_rejected: List[dict] = []

    for tier in tiers:
        index_list = write_tier_kb(tier, _KB_DIR)
        kb_text = (_KB_DIR / tier.file).read_text(encoding="utf-8")
        edge_index = {(e["a_name"], e["b_name"]): e for e in index_list}
        print(f"\n=== {tier.key}: {len(tier.graph.stations)} stations, "
              f"{len(tier.graph.edges)} edges -> {tier.file}")
        if tier.stranded:
            print(f"    stranded by the cut: {', '.join(tier.stranded)}")

        emitted, rejected = generate_configs(
            tier, kb_text, edge_index, net.zones, solve_ground_truth,
            args.gt_timeout, args.configs_per_tier, args.goals_per_cell)
        all_rows.extend(emitted)
        all_rejected.extend(rejected)

        gt_times = [r["gt_solve_time_s"] for r in emitted if r["gt_solve_time_s"] is not None]
        if emitted:
            print(f"    emitted {len(emitted)} configs; GT time "
                  f"min/median/max = {min(gt_times):.1f}/"
                  f"{sorted(gt_times)[len(gt_times) // 2]:.1f}/{max(gt_times):.1f}s")
        else:
            print(f"    NO CONFIGS EMITTED - tier is GT-infeasible with the current "
                  f"solver at a {args.gt_timeout:g}s budget")
            all_rejected.append({
                "tier": tier.key, "kb": tier.file, "config_name": "",
                "gates_passed": "", "gate_failures": "TIER GT-INFEASIBLE",
            })
        print(f"    rejected {len(rejected)} candidate(s) by the gates")

    manifest = _PROJECT_ROOT / "benchmark_manifest.csv"
    write_manifest(all_rows + all_rejected, manifest)
    print(f"\nwrote {manifest} ({len(all_rows)} configs, {len(all_rejected)} rejected rows)")
    write_station_atoms(graph, net, _PROJECT_ROOT / "station_atoms.csv")
    print(f"wrote {_PROJECT_ROOT / 'station_atoms.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
