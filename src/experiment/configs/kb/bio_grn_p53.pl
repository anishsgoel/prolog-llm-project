1. regulates(atm, tp53). # regulates/2 = DIRECTED REGULATORY EDGE: regulates(A, B) means transcription factor / kinase A directly regulates the activity or expression of B (sign ignored). This is the only base relation; indirect regulation and feedback are derived. Human p53 stress-response network.
2. regulates(tp53, mdm2).
3. regulates(mdm2, tp53).
4. regulates(tp53, cdkn1a).
5. regulates(tp53, bax).
6. regulates(tp53, gadd45a).
7. regulates(atm, chek2).
8. regulates(chek2, tp53).
9. influences(X, Y) :- regulates(X, Y). # influences/2 = X regulates Y directly or through a cascade (transitive closure of regulates/2).
10. influences(X, Z) :- regulates(X, Y), influences(Y, Z).
11. feedback(X) :- influences(X, X). # feedback/1 = X lies on a regulatory feedback loop (it influences itself through a cycle).