# Notation for the CDBEkP NP-Completeness Proof

Local notation used in the CDBEkP NP-completeness reduction from Set Cover. These symbols are
specific to this proof and are **not** part of the global knowledge base notation.

---

| Symbol                       | Name                     | Description                                                          | First Used In              |
| :--------------------------- | :----------------------- | :------------------------------------------------------------------- | :------------------------- |
| $W$                          | Set Cover universe       | Finite universe of objects in the Set Cover problem                  | prop-set-cover-problem     |
| $\mathcal{S}$                | Set Cover family         | Family of subsets of $W$ whose union equals $W$                      | prop-set-cover-problem     |
| $\mathcal{C}$                | Set Cover solution       | Subfamily of $\mathcal{S}$ covering $W$                              | prop-set-cover-problem     |
| $\omega$                     | Set Cover element        | Generic element of the universe $W$                                  | prop-set-cover-problem     |
| $\mathbb{A}_{\mathcal{S}}$   | Transformed decision table | Decision table constructed from $(W, \mathcal{S})$                 | prop-set-cover-construction |
| $U_{\mathcal{S}}$            | Transformed objects      | $\{u_\omega \mid \omega \in W\}$ in the transformed table            | prop-set-cover-construction |
| $A_{\mathcal{S}}$            | Transformed attributes   | $\{a_{S_i} \mid S_i \in \mathcal{S}\}$                               | prop-set-cover-construction |
| $d_{\mathcal{S}}$            | Transformed decision     | Decision attribute; $1$ on $u_*$ only                                | prop-set-cover-construction |
| $u_*$                        | Special object           | Additional object with decision value $1$                            | prop-set-cover-construction |
| $u_\omega$                   | Object from universe     | Object in the transformed table for $\omega \in W$                   | prop-set-cover-construction |
