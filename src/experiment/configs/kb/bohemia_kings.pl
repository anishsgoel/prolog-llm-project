1. parentof(premysl_otakar_i, wenceslaus_i). # parentof/2 = DIRECT PARENT. parentof(P, C) means person P is a biological parent of person C. This is the only base genealogical relation; every other relation is derived from it. Simplified genealogy of the Bohemian kings (Premyslid and Luxembourg dynasties).
2. parentof(premysl_otakar_i, vladislaus).
3. parentof(wenceslaus_i, premysl_otakar_ii).
4. parentof(wenceslaus_i, bozena).
5. parentof(premysl_otakar_ii, wenceslaus_ii).
6. parentof(vladislaus, agnes).
7. parentof(wenceslaus_ii, wenceslaus_iii).
8. parentof(wenceslaus_ii, elizabeth_premyslid).
9. parentof(bozena, otto_of_brandenburg).
10. parentof(john_of_luxembourg, charles_iv).
11. parentof(elizabeth_premyslid, charles_iv).
12. parentof(charles_iv, wenceslaus_iv).
13. parentof(charles_iv, sigismund).
14. king(premysl_otakar_i). # king/1 = person who reigned as King of Bohemia.
15. king(wenceslaus_i).
16. king(premysl_otakar_ii).
17. king(wenceslaus_ii).
18. king(wenceslaus_iii).
19. king(john_of_luxembourg).
20. king(charles_iv).
21. king(wenceslaus_iv).
22. king(sigismund).
23. child(C, P) :- parentof(P, C). # child/2 = child(C, P) means C is a child of P (the inverse of parentof/2).
24. ancestor(X, Y) :- parentof(X, Y). # ancestor/2 = ancestor(X, Y) means X is an ancestor of Y; transitive closure of parentof/2 (parent, grandparent, great-grandparent, ...).
25. ancestor(X, Z) :- parentof(X, Y), ancestor(Y, Z).
26. sibling(X, Y) :- parentof(P, X), parentof(P, Y). # sibling/2 = sibling(X, Y) means X and Y share a common parent P.
27. grandparent(X, Z) :- parentof(X, Y), parentof(Y, Z). # grandparent/2 = grandparent(X, Z) means X is a grandparent of Z.
28. cousin(X, Y) :- parentof(A, X), parentof(B, Y), sibling(A, B). # cousin/2 = cousin(X, Y) means the parent of X and the parent of Y are siblings (first cousins).
29. uncle(U, N) :- parentof(P, N), sibling(U, P). # uncle/2 = uncle(U, N) means U is a sibling of one of N's parents (uncle or aunt).
30. royal_descendant(X) :- king(A), ancestor(A, X). # royal_descendant/1 = person who descends from someone who was a King of Bohemia.