1. connected(union_square, 14th_street). # connected/2 = ADJACENT STOPS ONLY. Use only when two stations are immediate neighbors on the same line. Do NOT add shortcut edges that skip intermediate stations.  The stations are connected in New York underground.
2. connected(14th_street, 23rd_street).
3. connected(23rd_street, 34th_street).
4. connected(34th_street, times_square).
5. connected(times_square, 42nd_street).
6. connected(42nd_street, grand_central).
7. connected(grand_central, bryant_park).
8. reachable(X, Y) :- connected(X, Y). # reachable/2 = PATH EXISTS. One-hop reachable comes only from connected/2.
9. reachable(X, Z) :- connected(X, Y), reachable(Y, Z). # reachable/2 is the transitive closure of connected/2 (multi-hop path following only adjacent edges).