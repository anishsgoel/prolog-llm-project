1. reaction(glucose, g6p). # reaction/2 = DIRECTED METABOLIC CONVERSION: reaction(S, P) means substrate S is converted to product P by one enzymatic step. This is the only base relation; metabolite reachability is derived. Canonical glycolysis (each step is one enzyme; cofactors omitted).
2. reaction(g6p, f6p).
3. reaction(f6p, f16bp).
4. reaction(f16bp, g3p).
5. reaction(f16bp, dhap).
6. reaction(dhap, g3p).
7. reaction(g3p, bpg13).
8. reaction(bpg13, pg3).
9. reaction(pg3, pg2).
10. reaction(pg2, pep).
11. reaction(pep, pyruvate).
12. produces(X, Y) :- reaction(X, Y). # produces/2 = metabolite X can be converted to metabolite Y through a sequence of reactions (transitive closure of reaction/2).
13. produces(X, Z) :- reaction(X, Y), produces(Y, Z).