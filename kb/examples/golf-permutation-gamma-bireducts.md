---
id: ex-golf-permutation-gamma-bireducts
type: example
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [example, golf, bireducts, algorithms, ordering]
requires:
  [concept-decision-table,
   concept-gamma-decision-bireduct,
   prop-gamma-decision-bireduct-ordering]
see_also:
  [ex-golf-permutation-bireducts,
   ex-golf-all-bireducts,
   prop-gamma-decision-bireduct-ordering]
source: src-thesis-phd
---

# Golf Dataset -- Gamma-Decision Bireducts from the Ordering Algorithm

The same fifteen object permutations as in the standard decision bireduct example, processed by the
$\gamma$-decision bireduct ordering algorithm. Gamma-bireducts differ from ordinary bireducts in
that the functional dependency $\Rrightarrow^{\gamma}$ requires discernibility against the entire
universe $U$, not just the objects in $X$.

| Permutation | $\gamma$-decision bireduct $(X, B)$ |
|:------------|:-------------------------------------|
| $15(O), 8, 18(W), 1, 4, 7, 2, 14, 10, 12, 9, 16(T), 6, 3, 13, 5, 11, 17(H)$ | $(\{10,11,13\}, \{T,H\})$ |
| $17(H), 13, 16(T), 8, 18(W), 6, 11, 3, 14, 10, 15(O), 5, 7, 9, 2, 1, 4, 12$ | $(\{3,7,12,13\}, \{O\})$ |
| $3, 8, 16(T), 1, 18(W), 11, 9, 15(O), 14, 12, 6, 4, 7, 17(H), 10, 13, 2, 5$ | $(\{1,2,3,7,8,9,11,12,13\}, \{O,H\})$ |
| $2, 13, 5, 14, 11, 7, 12, 4, 3, 1, 9, 6, 8, 10, 17(H), 15(O), 18(W), 16(T)$ | $(\{1,\ldots,14\}, \{O,T,W\})$ |
| $9, 4, 12, 14, 1, 8, 7, 3, 10, 13, 6, 11, 2, 5, 18(W), 16(T), 17(H), 15(O)$ | $(\{1,\ldots,14\}, \{O,H,W\})$ |
| $11, 15(O), 2, 17(H), 1, 10, 5, 7, 9, 8, 3, 13, 16(T), 6, 14, 12, 4, 18(W)$ | $(\{2,5,9,10,11,13\}, \{T,H,W\})$ |
| $16(T), 2, 5, 17(H), 10, 11, 18(W), 14, 1, 12, 7, 9, 13, 6, 4, 8, 3, 15(O)$ | $(\{1,\ldots,14\}, \{O,H,W\})$ |
| $18(W), 6, 17(H), 15(O), 5, 8, 4, 7, 3, 2, 10, 9, 12, 11, 13, 14, 1, 16(T)$ | $(\emptyset, \emptyset)$ |
| $15(O), 2, 3, 13, 1, 17(H), 4, 16(T), 18(W), 6, 12, 14, 5, 8, 9, 10, 11, 7$ | $(\{2,5,9,10,11,13\}, \{T,H,W\})$ |
| $15(O), 17(H), 14, 1, 10, 7, 4, 3, 12, 13, 5, 18(W), 9, 16(T), 11, 8, 2, 6$ | $(\{2,5,9\}, \{T,W\})$ |
| $6, 5, 10, 9, 17(H), 15(O), 12, 16(T), 8, 18(W), 4, 2, 13, 3, 7, 1, 14, 11$ | $(\{1,\ldots,14\}, \{O,T,W\})$ |
| $11, 14, 9, 13, 3, 7, 8, 2, 5, 1, 12, 18(W), 6, 4, 10, 17(H), 15(O), 16(T)$ | $(\{1,\ldots,14\}, \{O,T,W\})$ |
| $13, 8, 6, 17(H), 7, 18(W), 9, 16(T), 5, 3, 4, 12, 15(O), 2, 10, 14, 11, 1$ | $(\{1,\ldots,14\}, \{O,T,W\})$ |
| $9, 17(H), 2, 4, 6, 13, 14, 7, 16(T), 11, 10, 15(O), 18(W), 3, 5, 1, 8, 12$ | $(\{1,\ldots,14\}, \{O,T,W\})$ |
| $18(W), 5, 3, 15(O), 12, 4, 16(T), 17(H), 7, 2, 13, 11, 10, 6, 8, 1, 14, 9$ | $(\{3,7,12,13\}, \{O\})$ |

## Remarks

Comparing with the standard bireduct results, $\gamma$-bireducts tend to have smaller object sets
(stronger discernibility requirement) but often share the same attribute subsets. The empty
bireduct $(\emptyset, \emptyset)$ occurs for one permutation (index 8) where the algorithm could
not find any non-empty $\gamma$-bireduct in the current configuration.

The fact that four different permutations produce $(\{1,\ldots,14\}, \{O,T,W\})$ (indices 4, 11,
12, 13) shows that $(\{O,T,W\})$ is a $\gamma$-decision reduct for the full universe. Similarly,
$(\{1,\ldots,14\}, \{O,H,W\})$ appears twice, confirming $\{O,H,W\}$ as another such reduct.
