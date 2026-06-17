---
id: prop-indiscernibility-equivalence-relation
type: proposition
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, indiscernibility]
requires: [concept-decision-table, concept-indiscernibility]
see_also:
  [concept-approximations,
   concept-consistency,
   concept-positive-region,
   concept-decision-reduct]
source: src-thesis-phd
---

# Indiscernibility is an Equivalence Relation

The indiscernibility relation $IND(B)$ is an equivalence relation. This fundamental property enables
the partition of the universe $U$ into equivalence classes, which is the basis of all rough set
constructions.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ and an attribute subset $B \subseteq A \cup \{d\}$ be given.
The indiscernibility relation $IND(B)$ is an equivalence relation on $U$.

## Proof

Recall from the [definition of indiscernibility](../concepts/indiscernibility.md) that for
$u, u' \in U$:

$$
u \; IND(B) \; u' \;\Longleftrightarrow\; \forall_{a \in B}\; a(u) = a(u')
$$

We verify the three defining properties of an equivalence relation.

### Reflexivity

For any $u \in U$, the equality $a(u) = a(u)$ holds trivially for every $a \in B$. Therefore
$u \; IND(B) \; u$ for all $u \in U$.

### Symmetry

Suppose $u \; IND(B) \; u'$. Then $\forall_{a \in B}\; a(u) = a(u')$. By symmetry of equality,
$\forall_{a \in B}\; a(u') = a(u)$, which implies $u' \; IND(B) \; u$.

### Transitivity

Suppose $u \; IND(B) \; u'$ and $u' \; IND(B) \; u''$. Then $\forall_{a \in B}\; a(u) = a(u')$ and
$\forall_{a \in B}\; a(u') = a(u'')$. By transitivity of equality,
$\forall_{a \in B}\; a(u) = a(u'')$, which implies $u \; IND(B) \; u''$.

## Consequences

Since $IND(B)$ is an equivalence relation, it partitions $U$ into a quotient set $U/B$ (also denoted
$U/IND(B)$). Each equivalence class $[u]_B = \{u' \in U : u \; IND(B) \; u'\}$ is a maximal set of
pairwise indiscernible objects.

This partition is the foundational structure upon which [approximations](../concepts/approximations.md),
the [positive region](../concepts/positive-region.md), and [decision
reducts](../concepts/decision-reduct.md) are built.
