1. interacts(brca1, bard1). # interacts/2 = DIRECT PHYSICAL PROTEIN-PROTEIN INTERACTION (binary, experimentally observed binding). Symmetric. This is the only base relation; co-complex membership is derived from it. Human DNA double-strand-break / homologous-recombination repair machinery.
2. interacts(brca1, palb2).
3. interacts(palb2, brca2).
4. interacts(brca2, rad51).
5. interacts(rad51, rad54).
6. interacts(mre11, rad50).
7. interacts(rad50, nbn).
8. interacts(nbn, atm).
9. interacts(atm, brca1).
10. interacts(Y, X) :- interacts(X, Y). # PPIs are undirected: if X binds Y then Y binds X.
11. co_complex(X, Y) :- interacts(X, Y). # co_complex/2 = X and Y belong to the same physical complex / repair module (connected component of the interaction graph).
12. co_complex(X, Z) :- interacts(X, Y), co_complex(Y, Z).