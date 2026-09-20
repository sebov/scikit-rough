# Notation for the CDBEkP NP-Hardness Proof

Local notation used in the CDBEkP NP-hardness reduction from Set Cover. These symbols are
specific to this proof and are **not** part of the global knowledge base notation.

---

| Symbol                       | Name                     | Description                                                          | First Used In              |
| :--------------------------- | :----------------------- | :------------------------------------------------------------------- | :------------------------- |
| $W$                          | Set Cover universe       | Finite universe of objects in the Set Cover problem                  | prop-set-cover-problem     |
| $\mathcal{S}$                | Set Cover family         | Family of subsets of $W$ whose union equals $W$                      | prop-set-cover-problem     |
| $\mathcal{C}$                | Set Cover solution       | Subfamily of $\mathcal{S}$ covering $W$                              | prop-set-cover-problem     |
| $\omega$                     | Set Cover element        | Generic element of the universe $W$                                  | prop-set-cover-problem     |
| $\mathcal{I}$                | Set Cover instance       | Instance $(W, \mathcal{S})$ of the Set Cover problem; indexes the transformed structures | prop-set-cover-construction |
| $\mathbb{A}_{\mathcal{I}}$   | Transformed decision table | Decision table constructed from the instance $\mathcal{I} = (W, \mathcal{S})$ | prop-set-cover-construction |
| $U_{\mathcal{I}}$            | Transformed objects      | $\{u_\omega \mid \omega \in W\}$ in the transformed table            | prop-set-cover-construction |
| $A_{\mathcal{I}}$            | Transformed attributes   | $\{a_{S_i} \mid S_i \in \mathcal{S}\}$                               | prop-set-cover-construction |
| $d_{\mathcal{I}}$            | Transformed decision     | Decision attribute; $1$ on $u_*$ only                                | prop-set-cover-construction |
| $u_*$                        | Special object           | Additional object with decision value $1$                            | prop-set-cover-construction |
| $u_\omega$                   | Object from universe     | Object in the transformed table for $\omega \in W$                   | prop-set-cover-construction |
| $\mathcal{K}$                | K-multiset               | Multiset of $(\{u_*\}, \emptyset)$ bireducts with multiplicity $K$   | prop-0-1-bireduct-ensemble-decomposition |
| $\mathcal{L}$                | L-multiset               | Multiset of $(U_{\mathcal{I}}, \emptyset)$ bireducts with multiplicity $L$ | prop-0-1-bireduct-ensemble-decomposition |
| $\mathcal{M}$                | M-multiset               | Multiset of single-attribute bireducts                                | prop-0-1-bireduct-ensemble-decomposition |
