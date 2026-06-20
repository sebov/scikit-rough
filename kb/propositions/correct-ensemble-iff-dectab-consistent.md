---
id: prop-correct-ensemble-iff-dectab-consistent
type: proposition
status: complete
created: 2026-06-20
updated: 2026-06-20
tags: [bireducts, ensemble, consistency, core]
requires:
  - concept-consistency
  - concept-decision-bireduct
  - concept-bireduct-ensemble
  - concept-decision-reduct
  - prop-decision-reduct-iff-bireduct
see_also:
  - prop-ensemble-np-hard
  - prop-decision-reduct-iff-bireduct
source: src-reduct-matrix-ensembles
---

# Correct Ensemble Exists iff Decision Table is Consistent

A correct ensemble of decision bireducts exists for a decision table if and only if that table is
consistent. This provides the fundamental link between ensemble existence and the classical
consistency condition.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be a decision table. There exists a correct ensemble of
decision bireducts for $\mathbb{A}$ if and only if the decision table is consistent.

## Proof

The proof has two directions.

### ($\Rightarrow$) Correct ensemble implies consistency

Assume, for contradiction, that there exists a correct ensemble

$$
\mathcal{B} = \{(X_1, B_1), \ldots, (X_m, B_m)\}
$$

of decision bireducts for $\mathbb{A}$, but $\mathbb{A}$ is inconsistent. By inconsistency, there
exists a pair of objects $u, u' \in U$ that are indiscernible by all attributes in $A$ -- that is,
$\forall_{a \in A}\; a(u) = a(u')$ -- while having different decision values, $d(u) \neq d(u')$.

For each bireduct $(X_i, B_i) \in \mathcal{B}$, we have $B_i \subseteq A$. Since $u$ and $u'$ are
indiscernible on $A$, they are also indiscernible on $B_i$. By definition of a decision bireduct,
$B_i \Rrightarrow_{X_i} d$: any two objects in $X_i$ that are $B_i$-indiscernible must have the same
decision value. Since $u$ and $u'$ are $B_i$-indiscernible but $d(u) \neq d(u')$, they cannot both
belong to $X_i$.

Since $\mathcal{B}$ is correct, by definition:

$$
cov_{\mathbb{A},\mathcal{B}}(u) > \frac{|\mathcal{B}|}{2}
$$

Let $k = cov_{\mathbb{A},\mathcal{B}}(u)$, the number of bireducts covering $u$. Then $u'$ is
covered by at most the remaining components -- those that do not cover $u$:

$$
cov_{\mathbb{A},\mathcal{B}}(u') \leq |\mathcal{B}| - k < |\mathcal{B}| - \frac{|\mathcal{B}|}{2}
= \frac{|\mathcal{B}|}{2}
$$

This contradicts the requirement that $\mathcal{B}$ is correct (every object must be covered by more
than half of the ensemble components). Therefore, $\mathbb{A}$ must be consistent.

### ($\Leftarrow$) Consistency implies existence of a correct ensemble

If $\mathbb{A}$ is consistent, then $A \Rrightarrow_U d$. By definition of a decision reduct, there
exists an irreducible subset $B \subseteq A$ such that $B \Rrightarrow_U d$. By
[prop-decision-reduct-iff-bireduct](decision-reduct-iff-bireduct.md), $(U, B)$ is a decision
bireduct for $\mathbb{A}$.

Construct the singleton ensemble $\mathcal{B} = \{(U, B)\}$. For every object $u \in U$:

$$
cov_{\mathbb{A},\mathcal{B}}(u) = 1 > \frac{1}{2} = \frac{|\mathcal{B}|}{2}
$$

Thus $\mathcal{B}$ satisfies the correctness condition. A correct ensemble exists.

## Consequences

This proposition provides the necessary and sufficient condition for the mere existence of a correct
ensemble. For consistent tables, a correct ensemble always exists (trivially, the singleton ensemble
built from any decision reduct). For inconsistent tables, no correct ensemble can exist using only
decision bireducts as components; ensemble-based approaches for inconsistent tables must employ
extensions such as $\gamma$-decision bireducts or $\varepsilon$-decision bireducts.

This result also underpins the NP-completeness and NP-hardness reductions in the ensemble complexity
analysis: the starting point for the Set Cover reduction is a consistent table
$\mathbb{A}_{\mathcal{S}}$, for which the existence of a correct ensemble is guaranteed by this
proposition.

## Remarks

The forward direction ($\Rightarrow$) uses only the coverage counting property of correct ensembles
and the fact that bireduct attribute subsets are subsets of $A$. It does not rely on any specific
simplicity measure.

The backward direction ($\Leftarrow$) constructs the simplest possible correct ensemble: a single
bireduct. This is always simpler than any multi-bireduct ensemble, establishing that consistent
tables always admit a correct ensemble with minimal complexity under any reasonable simplicity
measure.
