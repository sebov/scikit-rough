---
id: concept-discernibility-measure
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, evaluation, reduction]
requires: [concept-decision-table, concept-indiscernibility]
see_also: [concept-decision-reduct, concept-approximate-decision-reduct]
source: tmp/phd/thesis.tex
---

# Discernibility Measure

The discernibility measure quantifies how many pairs of objects with different decision values are
discerned by a given attribute subset. It serves as one of the evaluation functions for approximate
decision reducts.

## Definition

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. The discernibility measure
$disc_{\mathbb{A}} : 2^A \to \mathbb{N}$ is defined for $B \subseteq A$ as:

$$
disc_{\mathbb{A}}(B) = \lvert\{(u, u') \in U \times U :
  u \; DIS(B) \; u' \land d(u) \neq d(u')\}\rvert
$$

## Remarks

A normalized variant $disc\_ratio(B) \in [0, 1]$ can be obtained by dividing by the total number of
object pairs with different decisions. Both measures are nondecreasing and monotone with respect to
set inclusion, making them suitable as the evaluation function $F$ in the framework of
[approximate decision reducts](../concepts/approximate-decision-reduct.md).
