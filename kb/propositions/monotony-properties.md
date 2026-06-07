---
id: prop-monotony-properties
type: proposition
status: complete
created: 2026-06-05
updated: 2026-06-05
tags: [core, bireducts]
requires:
  [concept-decision-table,
   concept-decision-bireduct]
see_also:
  [concept-gamma-decision-bireduct,
   prop-decision-reduct-iff-bireduct]
source: src-thesis-phd
---

# Monotonicity Properties of Inexact Functional Dependency

The inexact functional dependency $B \Rrightarrow_X d$ is monotone with respect to both attribute
addition and object removal. These properties underpin the irreducibility and maximality conditions
in the decision bireduct definition.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. The following two monotony properties hold:

1. Let $B \subseteq B' \subseteq A$ and $X \subseteq U$ be given. If $B \Rrightarrow_X d$, then
   $B' \Rrightarrow_X d$.
2. Let $B \subseteq A$ and $X' \subseteq X \subseteq U$ be given. If $B \Rrightarrow_X d$, then
   $B \Rrightarrow_{X'} d$.

## Proof

**(1.)** Since there is $B \Rrightarrow_X d$, we know that $B$ discerns all pairs $u_i, u_j \in X$
for which $d(u_i) \neq d(u_j)$. It means that for each such pair there exists $a \in B$ that
discerns the objects, i.e., $a(u_i) \neq a(u_j)$. As $B \subseteq B'$, the same fact holds for $B'$
and therefore $B' \Rrightarrow_X d$.

**(2.)** We know that $B$ discerns all pairs $u_i, u_j \in X$ such that $d(u_i) \neq d(u_j)$. As
$X' \subseteq X$, the same fact holds for $X'$ which means that $B \Rrightarrow_{X'} d$.

## Consequences

Property (1) justifies the attribute irreducibility condition: if a proper subset $B' \subsetneq B$
already satisfies $B' \Rrightarrow_X d$, then $B$ is not minimal. Property (2) justifies the object
maximality condition: if a proper superset $X' \supsetneq X$ satisfies $B \Rrightarrow_{X'} d$, then
$X$ is not maximal.
