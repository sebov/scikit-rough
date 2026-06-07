---
id: prop-gamma-decision-bireduct-ordering
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [bireducts, algorithms, ordering, positive-region]
requires:
  [concept-decision-table,
   concept-gamma-decision-bireduct,
   prop-gamma-monotony-properties]
see_also:
  [prop-decision-bireduct-ordering,
   concept-gamma-decision-bireduct,
   prop-gamma-monotony-properties]
source: src-thesis-phd
---

# Correctness of the Gamma Decision Bireduct Ordering Algorithm

The gamma-decision bireduct ordering algorithm always produces a valid $\gamma$-decision bireduct,
and every $\gamma$-decision bireduct is achievable by some choice of input permutation. The proof
is analogous to that of the ordinary decision bireduct ordering algorithm, with the standard
functional dependency $B \Rrightarrow_X d$ replaced by the gamma functional dependency $B
\Rrightarrow^{\gamma}_X d$.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. For each permutation $\sigma : \{1, \ldots, |U| +
|A|\} \rightarrow \{1, \ldots, |U| + |A|\}$ the final outcome $(X_{|U|+|A|}, B_{|U|+|A|})$ of the
$\gamma$-decision bireduct ordering algorithm is a $\gamma$-decision bireduct. Moreover, for each
$\gamma$-decision bireduct $(X, B)$ there exists an input $\sigma$ for which the algorithm's output
equals $(X, B)$.

## Background

The $\gamma$-ordering algorithm is identical to the standard ordering algorithm except that the
functional dependency check uses $B \Rrightarrow^{\gamma}_X d$ instead of $B \Rrightarrow_X d$.
Specifically, when considering adding an object, the algorithm checks $B_i
\Rrightarrow^{\gamma}_{X_i \cup \{u\}} d$; when considering removing an attribute, it checks
$B_i \setminus \{a\} \Rrightarrow^{\gamma}_{X_i} d$.

## Proof

The proof is analogous to the proof of `prop-decision-bireduct-ordering`, with two adaptations.

**Part 1 (output is a bireduct).** The initial state $(X_0, B_0) = (\emptyset, A)$ satisfies $B_0
\Rrightarrow^{\gamma}_{X_0} d$ vacuously. Each step preserves the dependency by construction, so
$B_{|U|+|A|} \Rrightarrow^{\gamma}_{X_{|U|+|A|}} d$ holds at the end. Attribute irreducibility and
object maximality follow by the same contradiction arguments, using the gamma monotony properties
(Proposition `prop-gamma-monotony-properties`) in place of the standard monotony properties.

**Part 2 (every bireduct is achievable).** The same four-segment permutation construction is used:
(1) $A \setminus B$, (2) $X$, (3) $B$, (4) $U \setminus X$. The reasoning for each segment is
identical, using the $\gamma$-dependency and $\gamma$-bireduct condition in place of the standard
ones.
