1. targets(imatinib, abl1). # targets/2 = DRUG-PROTEIN inhibition/binding: targets(Drug, Protein) means the drug acts on that protein. A base relation of this heterogeneous drug-target-disease network.
2. targets(imatinib, kit).
3. targets(imatinib, pdgfra).
4. targets(dasatinib, abl1).
5. targets(dasatinib, src).
6. implicated(abl1, cml). # implicated/2 = PROTEIN-DISEASE association: implicated(Protein, Disease) means the protein is a causal driver of the disease. A base relation.
7. implicated(kit, gist).
8. implicated(pdgfra, gist).
9. implicated(src, colorectal_cancer).
10. interacts(kit, pdgfra). # interacts/2 = protein-protein interaction between related receptor tyrosine kinases.
11. treats(D, Dis) :- targets(D, P), implicated(P, Dis). # treats/2 = candidate therapeutic link: drug D may treat disease Dis because it acts on a protein driving Dis.
12. treats(D, Dis) :- targets(D, P), interacts(P, Q), implicated(Q, Dis).