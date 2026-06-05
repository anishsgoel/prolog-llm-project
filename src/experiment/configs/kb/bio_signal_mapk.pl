1. signals(egf, egfr). # signals/2 = DIRECTED ACTIVATING STEP in a signal-transduction cascade: signals(A, B) means active A directly activates/phosphorylates B. This is the only base relation; downstream propagation is derived. Canonical EGF -> MAPK/ERK pathway.
2. signals(egfr, grb2).
3. signals(grb2, sos1).
4. signals(sos1, kras).
5. signals(kras, braf).
6. signals(braf, map2k1).
7. signals(map2k1, mapk1).
8. signals(mapk1, elk1).
9. signals(mapk1, myc).
10. propagates(X, Y) :- signals(X, Y). # propagates/2 = the signal starting at X eventually reaches/activates Y (transitive closure of signals/2).
11. propagates(X, Z) :- signals(X, Y), propagates(Y, Z).