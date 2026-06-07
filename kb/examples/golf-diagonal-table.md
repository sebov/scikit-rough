---
id: ex-golf-diagonal-table
type: example
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [example, golf, bireducts, diagonal]
requires:
  [concept-decision-table,
   concept-decision-bireduct,
   prop-decision-table-diagonal]
see_also:
  [prop-decision-table-diagonal,
   ex-golf-all-bireducts,
   concept-decision-bireduct]
source: src-thesis-phd
---

# Golf Dataset -- Diagonal Table Transformation

The diagonal transformation augments the golf decision table with 14 additional binary attributes
$a^{\boxbslash}_1, \ldots, a^{\boxbslash}_{14}$, one per object. Each diagonal attribute
$a^{\boxbslash}_i$ has value 1 only on object $u_i$ and 0 elsewhere, uniquely identifying that
object. The original four conditional attributes ($O, T, H, W$) and the decision $d$ are preserved.

Decision bireducts of the original table correspond to standard decision reducts on this
diagonal-augmented table.

| | O | T | H | W | $a^{\boxbslash}_1$ | $a^{\boxbslash}_2$ | $a^{\boxbslash}_3$ | $a^{\boxbslash}_4$ | $a^{\boxbslash}_5$ | $a^{\boxbslash}_6$ | $a^{\boxbslash}_7$ | $a^{\boxbslash}_8$ | $a^{\boxbslash}_9$ | $a^{\boxbslash}_{10}$ | $a^{\boxbslash}_{11}$ | $a^{\boxbslash}_{12}$ | $a^{\boxbslash}_{13}$ | $a^{\boxbslash}_{14}$ | d |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:---|
| 1 | sunny | hot | high | weak | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no |
| 2 | sunny | hot | high | strong | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no |
| 3 | overcast | hot | high | weak | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes |
| 4 | rain | mild | high | weak | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes |
| 5 | rain | cool | normal | weak | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes |
| 6 | rain | cool | normal | strong | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no |
| 7 | overcast | cool | normal | strong | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes |
| 8 | sunny | mild | high | weak | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | no |
| 9 | sunny | cool | normal | weak | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | yes |
| 10 | rain | mild | normal | weak | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | yes |
| 11 | sunny | mild | normal | strong | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | yes |
| 12 | overcast | mild | high | strong | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | yes |
| 13 | overcast | hot | normal | weak | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | yes |
| 14 | rain | mild | high | strong | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | no |

## Remarks

The diagonal attributes $a^{\boxbslash}_i$ enforce that no two objects can be indiscernible in the
augmented table -- each object has a unique "fingerprint". This device transforms the bireduct
problem (where we jointly select objects and attributes) into a standard reduct problem (where we
only select attributes) on the augmented table. The diagonal attributes correspond to propositional
variables for objects in the Boolean formula characterization $\tau_{bi}$.
