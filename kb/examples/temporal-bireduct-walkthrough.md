---
id: ex-temporal-bireduct-walkthrough
type: example
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [example, temporal, bireducts, algorithms, streaming]
requires:
  [concept-decision-table,
   concept-decision-bireduct,
   concept-temporal-bireduct,
   prop-temporal-bireduct-computation]
see_also:
  [prop-temporal-bireduct-computation,
   concept-temporal-bireduct,
   ex-golf-all-bireducts]
source: tmp/phd/include/temporal_bireducts.tex
---

# Temporal Bireduct Computation Walkthrough

A step-by-step trace of the streaming buffer algorithm on the golf dataset for two choices of the
restart attribute set $A'$. The algorithm processes objects $u_1, \ldots, u_{14}$ sequentially,
maintaining a contiguous buffer $X$ and an attribute subset $B$. When a new object breaks the
dependency $B \Rrightarrow_X d$, the current $(X, B)$ is saved as a temporal bireduct and $B$ is
reset to $A'$.

## Case 1: $A' = \{O, T, H\}$

The restart attribute set excludes $W$ (Wind). Each object row below shows $O, T, H, d$ values.

| Step | Object arrives | $(X, B)$ after processing | Action |
|:-----|:---------------|:--------------------------|:-------|
| 1 | $u_1$: sunny, hot, high (no) | $(\{1\}, \emptyset)$ | add to buffer |
| 2 | $u_2$: sunny, hot, high (no) | $(\{1,2\}, \emptyset)$ | add to buffer |
| 3 | $u_3$: overcast, hot, high (yes) | $(\{1,2,3\}, \{O\})$ | reset $B=A'$, drop none, reduce to $\{O\}$; save $(\{1,2\}, \emptyset)$ |
| 4 | $u_4$: rain, mild, high (yes) | $(\{1,2,3,4\}, \{O\})$ | add to buffer |
| 5 | $u_5$: rain, cool, normal (yes) | $(\{1,2,3,4,5\}, \{O\})$ | add to buffer |
| 6 | $u_6$: rain, cool, normal (no) | $(\{6\}, \emptyset)$ | reset, drop $u_{1..5}$, reduce to $\emptyset$; save $(\{1,2,3,4,5\}, \{O\})$ |
| 7 | $u_7$: overcast, cool, normal (yes) | $(\{6,7\}, \{O\})$ | reset, drop none, reduce to $\{O\}$; save $(\{6\}, \emptyset)$ |
| 8 | $u_8$: sunny, mild, high (no) | $(\{6,7,8\}, \{O\})$ | add to buffer |
| 9 | $u_9$: sunny, cool, normal (yes) | $(\{6,7,8,9\}, \{O,H\})$ | reset, drop none, reduce to $\{O,H\}$; save $(\{6,7,8\}, \{O\})$ |
| 10 | $u_{10}$: rain, mild, normal (yes) | $(\{6,7,8,9,10\}, \{O,T\})$ | reset, drop none, reduce to $\{O,T\}$; save $(\{6,7,8,9\}, \{O,H\})$ |
| 11 | $u_{11}$: sunny, mild, normal (yes) | $(\{6,\ldots,11\}, \{O,T,H\})$ | reset, drop none, reduce to $\{O,T,H\}$; save $(\{6,7,8,9,10\}, \{O,T\})$ |
| 12 | $u_{12}$: overcast, mild, high (yes) | $(\{6,\ldots,12\}, \{O,T,H\})$ | add to buffer |
| 13 | $u_{13}$: overcast, hot, normal (yes) | $(\{6,\ldots,13\}, \{O,T,H\})$ | add to buffer |
| 14 | $u_{14}$: rain, mild, high (no) | $(\{6,\ldots,14\}, \{O,T,H\})$ | add to buffer |

**Saved temporal bireducts:** $(\{1,2\}, \emptyset)$, $(\{1,2,3,4,5\}, \{O\})$, $(\{6\}, \emptyset)$,
$(\{6,7,8\}, \{O\})$, $(\{6,7,8,9\}, \{O,H\})$, $(\{6,7,8,9,10\}, \{O,T\})$.

## Case 2: $A' = \{T, H, W\}$

The restart attribute set excludes $O$ (Outlook). Each object row below shows $T, H, W, d$ values.

| Step | Object arrives | $(X, B)$ after processing | Action |
|:-----|:---------------|:--------------------------|:-------|
| 1 | $u_1$: hot, high, weak (no) | $(\{1\}, \emptyset)$ | add to buffer |
| 2 | $u_2$: hot, high, strong (no) | $(\{1,2\}, \emptyset)$ | save $(\{1,2\}, \emptyset)$ |
| 3 | $u_3$: hot, high, weak (yes) | $(\{2,3\}, \{W\})$ | reset $B=A'$, drop $u_1$, reduce |
| 4 | $u_4$: mild, high, weak (yes) | $(\{2,3,4\}, \{W\})$ | add to buffer |
| 5 | $u_5$: cool, normal, weak (yes) | $(\{2,3,4,5\}, \{W\})$ | add to buffer |
| 6 | $u_6$: cool, normal, strong (no) | $(\{2,3,4,5,6\}, \{W\})$ | save $(\{2,3,4,5,6\}, \{W\})$ |
| 7 | $u_7$: cool, normal, strong (yes) | $(\{7\}, \emptyset)$ | reset, save $(\{7\}, \emptyset)$ |
| 8 | $u_8$: mild, high, weak (no) | $(\{7,8\}, \{W\})$ | save $(\{7,8\}, \{W\})$ |
| 9 | $u_9$: cool, normal, weak (yes) | $(\{7,8,9\}, \{H\})$ | reset, reduce to $\{H\}$ |
| 10 | $u_{10}$: mild, normal, weak (yes) | $(\{7,8,9,10\}, \{H\})$ | add to buffer |
| 11 | $u_{11}$: mild, normal, strong (yes) | $(\{7,\ldots,11\}, \{H\})$ | save $(\{7,\ldots,11\}, \{H\})$ |
| 12 | $u_{12}$: mild, high, strong (yes) | $(\{7,\ldots,12\}, \{H,W\})$ | reset, reduce to $\{H,W\}$ |
| 13 | $u_{13}$: hot, normal, weak (yes) | $(\{7,\ldots,13\}, \{H,W\})$ | save $(\{7,\ldots,13\}, \{H,W\})$ |
| 14 | $u_{14}$: mild, high, strong (no) | $(\{13,14\}, \{W\})$ | reset, drop $u_{7,\ldots,12}$, reduce to $\{W\}$ |

**Saved temporal bireducts:** $(\{1,2\}, \emptyset)$, $(\{2,3,4,5,6\}, \{W\})$, $(\{7\},
\emptyset)$, $(\{7,8\}, \{W\})$, $(\{7,\ldots,11\}, \{H\})$, $(\{7,\ldots,13\}, \{H,W\})$.

## Remarks

The choice of $A'$ significantly affects which temporal bireducts are produced. Both cases
produce 6 bireducts but with different object ranges and attribute subsets. The two cases
demonstrate complementary restart strategies: one using the outlook-oriented subset
$\{O,T,H\}$, the other using $\{T,H,W\}$.

The saved pairs satisfy the temporal bireduct conditions: forward non-extendability (the next
object broke the dependency), backward non-extendability (oldest objects were removed in earlier
resets), attribute irreducibility (heuristic reduction), and functional dependency (by
construction after reset).
