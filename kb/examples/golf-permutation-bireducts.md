---
id: ex-golf-permutation-bireducts
type: example
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [example, golf, bireducts, algorithms, ordering]
requires:
  [concept-decision-table,
   concept-decision-bireduct,
   prop-decision-bireduct-ordering]
see_also:
  [ex-golf-permutation-gamma-bireducts,
   ex-golf-all-bireducts,
   prop-decision-bireduct-ordering]
source: src-thesis-phd
---

# Golf Dataset -- Decision Bireducts from the Ordering Algorithm

Fifteen different object permutations and the decision bireduct produced by the ordering algorithm
for each. The permutation lists objects $u_1, \ldots, u_{14}$ interleaved with four attributes
$O, T, H, W$ shown parenthetically at their positions. The algorithm processes objects in order;
when a new object breaks the functional dependency, the current bireduct is saved and $B$ is reset
to $A' = A$.

| Permutation | Decision bireduct $(X, B)$ |
|:------------|:---------------------------|
| $15(O), 8, 18(W), 1, 4, 7, 2, 14, 10, 12, 9, 16(T), 6, 3, 13, 5, 11, 17(H)$ | $(\{1,2,5,7,8,9,10,11,13,14\}, \{H\})$ |
| $17(H), 13, 16(T), 8, 18(W), 6, 11, 3, 14, 10, 15(O), 5, 7, 9, 2, 1, 4, 12$ | $(\{1,2,3,6,7,8,12,13,14\}, \{O\})$ |
| $3, 8, 16(T), 1, 18(W), 11, 9, 15(O), 14, 12, 6, 4, 7, 17(H), 10, 13, 2, 5$ | $(\{1,2,3,6,7,8,9,11,12,13,14\}, \{O,H\})$ |
| $2, 13, 5, 14, 11, 7, 12, 4, 3, 1, 9, 6, 8, 10, 17(H), 15(O), 18(W), 16(T)$ | $(\{1,\ldots,14\}, \{O,T,W\})$ |
| $9, 4, 12, 14, 1, 8, 7, 3, 10, 13, 6, 11, 2, 5, 18(W), 16(T), 17(H), 15(O)$ | $(\{1,\ldots,14\}, \{O,H,W\})$ |
| $11, 15(O), 2, 17(H), 1, 10, 5, 7, 9, 8, 3, 13, 16(T), 6, 14, 12, 4, 18(W)$ | $(\{1,2,4,5,7,9,10,11,12\}, \{T\})$ |
| $16(T), 2, 5, 17(H), 10, 11, 18(W), 14, 1, 12, 7, 9, 13, 6, 4, 8, 3, 15(O)$ | $(\{1,2,3,4,5,7,8,10,12,13\}, \{O\})$ |
| $18(W), 6, 17(H), 15(O), 5, 8, 4, 7, 3, 2, 10, 9, 12, 11, 13, 14, 1, 16(T)$ | $(\{3,6,8,13,14\}, \{T\})$ |
| $15(O), 2, 3, 13, 1, 17(H), 4, 16(T), 18(W), 6, 12, 14, 5, 8, 9, 10, 11, 7$ | $(\{2,3,4,5,6,9,10,13,14\}, \{W\})$ |
| $15(O), 17(H), 14, 1, 10, 7, 4, 3, 12, 13, 5, 18(W), 9, 16(T), 11, 8, 2, 6$ | $(\{1,2,4,5,7,9,10,14\}, \{T,W\})$ |
| $6, 5, 10, 9, 17(H), 15(O), 12, 16(T), 8, 18(W), 4, 2, 13, 3, 7, 1, 14, 11$ | $(\{2,3,4,5,6,9,10,11,12,13\}, \{T,W\})$ |
| $11, 14, 9, 13, 3, 7, 8, 2, 5, 1, 12, 18(W), 6, 4, 10, 17(H), 15(O), 16(T)$ | $(\{1,2,3,5,7,8,9,10,11,12,13,14\}, \{O,H\})$ |
| $13, 8, 6, 17(H), 7, 18(W), 9, 16(T), 5, 3, 4, 12, 15(O), 2, 10, 14, 11, 1$ | $(\{1,2,3,4,6,7,8,9,10,12,13\}, \{O,T\})$ |
| $9, 17(H), 2, 4, 6, 13, 14, 7, 16(T), 11, 10, 15(O), 18(W), 3, 5, 1, 8, 12$ | $(\{2,3,4,5,6,7,9,10,12,13,14\}, \{O,W\})$ |
| $18(W), 5, 3, 15(O), 12, 4, 16(T), 17(H), 7, 2, 13, 11, 10, 6, 8, 1, 14, 9$ | $(\{3,4,5,7,9,10,11,12,13\}, \emptyset)$ |

## Remarks

The permutations use a numbering scheme where objects $u_1, \ldots, u_{14}$ are listed by index and
attributes are written as: $15 = O$, $16 = T$, $17 = H$, $18 = W$. The algorithm includes
attributes in the permutation to control the order in which they are considered for the heuristic
reduction step.

Only bireducts actually saved by the algorithm are shown. Between saves, the buffer accumulates
objects until the dependency breaks. The algorithm always outputs a decision bireduct -- see
[Correctness of the Decision Bireduct Ordering Algorithm](../propositions/decision-bireduct-ordering.md).

Note that the bireduct $(\{3,4,5,7,9,10,11,12,13\}, \emptyset)$ shows the empty attribute set can
suffice for a subset where all objects share the same decision value.
