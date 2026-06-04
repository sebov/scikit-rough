---
tags: [rst, core]
related: [notation_and_symbols.md, definitions/decision_table.md, definitions/indiscernibility.md, definitions/consistency.md, definitions/approximations.md, definitions/reducts.md]
---

# Positive Region

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ and an attribute subset $B \subseteq A$ be
given. By the **positive region** $POS_B(d)$ we mean the subset of $U$ consisting of all objects
that can be uniquely classified to the decision classes using attributes in $B$:

$$
  POS_B(d) = \{\, u \in U : \forall_{u' \in [u]_B}\; d(u') = d(u) \,\}
$$

Equivalently, using equivalence classes induced by $B$:

$$
  POS_B(d) = \bigcup \{\, [u]_B \in U/B : \forall_{u, u' \in [u]_B}\; d(u) = d(u') \,\}
$$

That is, $POS_B(d)$ is the union of those $B$-induced equivalence classes within which all objects
share the same decision value.

## Relationship to Consistency

A decision table $\mathbb{A}$ is consistent if and only if the positive region with respect to the
full set of conditional attributes covers the entire universe:

$$
  \mathbb{A} \text{ is consistent} \iff POS_A(d) = U
$$

For inconsistent decision tables, extensions of the decision reduct notion can be formulated using
the positive region, generalized decision functions, or rough membership functions.

## Dependency Degree

The degree of dependency between an attribute subset $B \subseteq A$ and the decision $d$ is
expressed by the function $\gamma : 2^A \rightarrow [0, 1]$ defined as:

$$
  \gamma(B) = \frac{|POS_B(d)|}{|U|}
$$

A decision table $\mathbb{A}$ is consistent if and only if $\gamma(A) = 1$.
