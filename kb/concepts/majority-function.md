---
id: concept-majority-function
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, evaluation, reduction]
requires: [concept-decision-table, concept-indiscernibility, concept-approximate-decision-reduct]
see_also: [concept-relative-gain-function, concept-discernibility-measure, concept-epsilon-decision-bireduct]
source: src-thesis-phd
---

# Majority Function

The majority function $M$ models the accuracy of a rule-based classifier that, for each equivalence
class induced by a subset of attributes, points at the most frequent decision within that class.

## Definition

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. The majority function $M : 2^A \to [0, 1]$ is defined
for $B \subseteq A$ as:

$$
M(B) = \frac{1}{\lvert U \rvert}
       \sum_{E \in U/B}
       \max_{k = 1, \ldots, \lvert V_d \rvert}
       \lvert E \cap X^{\langle k \rangle} \rvert
$$

## Intuition

For each $B$-induced equivalence class $E$, the classifier predicts the most common decision value
among objects in $E$. The fraction of correctly classified objects under this scheme is $M(B)$. When
$M(B) = 1$, every equivalence class is homogeneous with respect to the decision.

## Role in Approximate Reducts

$M$ is a nondecreasing monotone function suitable as $F$ in the [approximate decision
reduct](../concepts/approximate-decision-reduct.md) framework. An
$M$-decision $\varepsilon$-reduct contains attributes sufficient for the majority classifier to
achieve accuracy at least $1 - \varepsilon$.

There is a close relationship between $M$-decision $\varepsilon$-reducts and [$\varepsilon$-decision
bireducts](../concepts/epsilon-decision-bireduct.md): the smallest $M$-decision $\varepsilon$-reduct
corresponds to the smallest attribute subset in an $\varepsilon$-decision bireduct.
