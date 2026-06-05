1. is_a(homologous_recombination, double_strand_break_repair). # is_a/2 = GENE ONTOLOGY SUBSUMPTION edge: is_a(Child, Parent) means the Child term is a more specific kind of the Parent term. Directed, acyclic. One of the base relations the program reasons over (the GO biological_process hierarchy).
2. is_a(double_strand_break_repair, dna_repair).
3. is_a(dna_repair, dna_metabolic_process).
4. is_a(dna_metabolic_process, metabolic_process).
5. is_a(metabolic_process, biological_process).
6. is_a(mismatch_repair, dna_repair).
7. annotated(brca1, homologous_recombination). # annotated/2 = direct curated GO annotation: gene G is annotated with term T.
8. annotated(rad51, homologous_recombination).
9. annotated(msh2, mismatch_repair).
10. subsumes(X, Z) :- is_a(X, Z). # subsumes/2 = term X is the same as or more specific than term Z (transitive closure of is_a/2).
11. subsumes(X, Z) :- is_a(X, Y), subsumes(Y, Z).
12. has_function(G, T2) :- annotated(G, T), subsumes(T, T2). # has_function/2 = GO true-path rule: a gene annotated to T also has every ancestor term T2 of T.