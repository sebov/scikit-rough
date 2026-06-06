---
id: prop-gamma-monotony-properties
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [core, bireducts, positive-region]
requires:
  [concept-decision-table,
   concept-gamma-decision-bireduct]
see_also:
  [prop-monotony-properties,
   concept-positive-region,
   concept-gamma-decision-reduct]
source: tmp/phd/thesis.tex
---

# Monotonicity Properties of Gamma Functional Dependency

The gamma functional dependency $B \Rrightarrow^{\gamma}_X d$ is monotone with respect to both
attribute addition and object removal. These properties underpin the irreducibility and maximality
conditions in the $\gamma$-decision bireduct definition.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. The following two monotony properties hold:

1. Let $X \subseteq U$ and $B \subseteq B' \subseteq A$ be given. If $B \Rrightarrow^{\gamma}_X d$,
   then $B' \Rrightarrow^{\gamma}_X d$.
2. Let $X' \subseteq X \subseteq U$ and $B \subseteq A$ be given. If $B \Rrightarrow^{\gamma}_X d$,
   then $B \Rrightarrow^{\gamma}_{X'} d$.

## Proof

The proof is analogous to the proof of
[prop-monotony-properties](monotony-properties.md). The gamma functional dependency
$B \Rrightarrow^{\gamma}_X d$ requires that $B$ discerns all pairs $u_i \in X$, $u_j \in U$ with
$d(u_i) \neq d(u_j)$.

**(1.)** If $B \Rrightarrow^{\gamma}_X d$, then for each pair $u_i \in X$, $u_j \in U$ with
$d(u_i) \neq d(u_j)$, there exists $a \in B$ such that $a(u_i) \neq a(u_j)$. As $B \subseteq B'$,
the same attribute $a \in B'$ discerns the pair, so $B' \Rrightarrow^{\gamma}_X d$.

**(2.)** If $B \Rrightarrow^{\gamma}_X d$, then for each pair $u_i \in X$, $u_j \in U$ with
$d(u_i) \neq d(u_j)$, there exists $a \in B$ such that $a(u_i) \neq a(u_j)$. As $X' \subseteq X$,
every $u_i \in X'$ is also in $X$, so the same discernibility holds for pairs from $X'$, meaning
$B \Rrightarrow^{\gamma}_{X'} d$.

## Consequences

Property (1) justifies the attribute irreducibility condition: if a proper subset $B' \subsetneq B$
already satisfies $B' \Rrightarrow^{\gamma}_X d$, then $B$ is not minimal. Property (2) justifies
the object maximality condition: if a proper superset $X' \supsetneq X$ satisfies
$B \Rrightarrow^{\gamma}_{X'} d$, then $X$ is not maximal.
