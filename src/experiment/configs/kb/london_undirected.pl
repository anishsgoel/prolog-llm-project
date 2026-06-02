1. connected(oxford_circus, piccadilly_circus). # connected/2 = ADJACENT STOPS ONLY. Use only when two stations are immediate neighbors on the same line. Do NOT add shortcut edges that skip intermediate stations.  The stations are connected by London Tube.
2. connected(piccadilly_circus, charing_cross).
3. connected(charing_cross, embankment).
4. connected(embankment, waterloo).
5. connected(waterloo, lamberth_north).
6. connected(lamberth_north, elephant_and_castle).
7. connected(regents_park, oxford_circus).
8. reachable(X, Y) :- connected(X, Y). # reachable/2 = PATH EXISTS. One-hop reachable comes only from connected/2.
9. reachable(X, Z) :- connected(X, Y), reachable(Y, Z). # reachable/2 is the transitive closure of connected/2 (multi-hop path following only adjacent edges).
10. connected(X, Y) :- connected(Y, X).