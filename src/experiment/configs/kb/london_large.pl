1. connected(paddington, edgware_road). # connected/2 = ADJACENT STOPS ONLY. Use only when two stations are immediate neighbours on bakerloo line. Do NOT add shortcut edges that skip intermediate stations. This is the London Underground Bakerloo line (with its northern stations), running from Harrow direction through Paddington down to Elephant & Castle.
2. connected(edgware_road, marylebone).
3. connected(marylebone, baker_street).
4. connected(baker_street, regents_park).
5. connected(regents_park, oxford_circus).
6. connected(oxford_circus, piccadilly_circus).
7. connected(piccadilly_circus, charing_cross).
8. connected(charing_cross, embankment).
9. connected(embankment, waterloo).
10. connected(waterloo, lambeth_north).
11. connected(lambeth_north, elephant_and_castle).
12. connected(warwick_avenue, paddington).
13. connected(maida_vale, warwick_avenue).
14. connected(kilburn_park, maida_vale).
15. connected(queens_park, kilburn_park).
16. connected(kensal_green, queens_park).
17. reachable(X, Y) :- connected(X, Y). # reachable/2 = PATH EXISTS. One-hop reachable comes only from connected/2.
18. reachable(X, Z) :- connected(X, Y), reachable(Y, Z). # reachable/2 is the transitive closure of connected/2 (multi-hop path following only adjacent edges).
19. connected(X, Y) :- connected(Y, X). # the tube is bidirectional: adjacency is symmetric.