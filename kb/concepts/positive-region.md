---
id: concept-positive-region
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, positive-region]
requires: [concept-decision-table, concept-indiscernibility, concept-approximations]
see_also: [concept-gamma-decision-reduct, concept-consistency]
source: tmp/phd/thesis.tex
---

# Positive Region

The positive region is the set of objects that can be uniquely classified to decision classes using a
given attribute subset. It is a central concept for extending decision reducts to inconsistent tables.

## Definition

Let $\mathbb{A} = (U, A \cup \{d\})$ and $B \subseteq A$ be given. The positive region $POS(B)$ (or
$POS_{\mathbb{A}}(B)$ when disambiguation is needed) is the subset of $U$ consisting of all objects
that can be uniquely classified to decision classes using attributes in $B$:

$$
POS(B) = \{u \in U : \forall_{u' \in [u]_B}\; d(u') = d(u)\}
$$

Equivalently, using $B$-induced equivalence classes:

$$
POS(B) = \bigcup\{E \in U/B : \forall_{u, u' \in E}\; d(u) = d(u')\}
$$

That is, $POS(B)$ is the union of those equivalence classes induced by $B$ within which all objects
share the same decision value.

## Degree of Dependency

The function $\gamma : 2^A \to [0, 1]$ expresses the degree of dependence between a subset of
attributes and the decision:

$$
\gamma(B) = \frac{\lvert POS(B) \rvert}{\lvert U \rvert}
$$

## Relationship to Consistency

A decision table $\mathbb{A}$ is consistent if and only if $POS(A) = U$, equivalently
$\gamma(A) = 1$.

## Remarks

The positive region is the foundation for [$\gamma$-decision
reducts](../concepts/gamma-decision-reduct.md). Objects in $POS(B)$ support deterministic decision
rules (confidence = 1), while objects outside $POS(B)$ correspond to conflicting indiscernibility
classes where deterministic classification is impossible with the given attributes.
