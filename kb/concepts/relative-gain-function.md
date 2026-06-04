---
id: concept-relative-gain-function
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, evaluation, reduction]
requires: [concept-decision-table, concept-indiscernibility, concept-approximate-decision-reduct]
see_also: [concept-majority-function, concept-discernibility-measure]
source: tmp/phd/thesis.tex
---

# Relative Gain Function

The relative gain function $R$ extends the classical rough set model with a Bayesian perspective.
For each equivalence class induced by a subset of attributes, it evaluates the decision class that
becomes maximally frequent relative to its overall occurrence in the data.

## Definition

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. The relative gain function $R : 2^A \to [0, 1]$ is
defined for $B \subseteq A$ as:

$$
R(B) = \frac{1}{\lvert V_d \rvert}
       \sum_{E \in U/B}
       \max_{k = 1, \ldots, \lvert V_d \rvert}
       \frac{\lvert E \cap X^{\langle k \rangle} \rvert}{\lvert X^{\langle k \rangle} \rvert}
$$

## Intuition

While the [majority function](../concepts/majority-function.md) considers absolute counts within each
equivalence class, $R$ normalizes by the overall size of each decision class. This makes $R$
sensitive to minority classes: a small equivalence class that entirely captures a rare decision class
contributes significantly to $R(B)$, whereas it would barely affect $M(B)$.

## Role in Approximate Reducts

$R$ is a nondecreasing monotone function suitable as $F$ in the [approximate decision
reduct](../concepts/approximate-decision-reduct.md) framework. Search algorithms for decision
bireducts tend to produce attribute subsets that correspond to $R$-decision $\varepsilon$-reducts
when objects from less frequent decision classes are added to $X$ relatively earlier.

## Remarks

For consistent tables, $M(A) = R(A) = 1$. For inconsistent tables, $M$ and $R$ may induce different
approximate reducts, reflecting their different treatment of class imbalance.
