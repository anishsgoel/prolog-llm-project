1. productof(malate, oxalocetate). # productof/2 = encodeds reactant and its immediate product in Krebs cycle
2. productof(oxalocetate, citrate).
3. productof(citrate, isocitrate).
4. productof(isocitrate, ketoglukarate).
5. productof(ketoglukarate, succinyl_coa).
6. productof(succinyl_coa, succinate).
7. productof(succinate, furmate).
8. productof(furmate, malate).
8. produces(X, Y) :- productof(X, Z), produces(Z, Y). # produces/2 = There is a sequence of reactions that X becomes Y.
9. cycle(X) :- produces(X, X). # cycle/1 Encodes that there is a cycle on which X can be transformed back to X by a chain of reactions.
10. produces(X, Y) :- productof(X, Y).