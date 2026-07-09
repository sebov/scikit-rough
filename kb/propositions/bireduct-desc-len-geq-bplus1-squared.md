---
id: prop-bireduct-desc-len-geq-bplus1-squared
type: proposition
status: complete
created: 2026-07-07
updated: 2026-07-07
tags: [bireducts, complexity]
requires:
  [prop-bireduct-desc-len-formula,
   prop-bireduct-equiv-classes-geq-bplus1]
see_also:
  [concept-bireduct-ensemble,
   concept-decision-bireduct]
source: "Slezak & Stawicki, 'Complexity of Searching for the Simplest Reduct Matrix Ensembles'
  (paper in preparation)"
---

# Bireduct Description Length Lower Bound

The description length of a decision bireduct is bounded below by the square of the number of
attributes plus one. This follows directly from the description length formula and the equivalence
classes bound.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be a decision table and $(X, B)$ be a decision bireduct for
$\mathbb{A}$. Then:

$$
BirDesc(X, B) \geq (|B| + 1)^2
$$

## Proof

This is a direct consequence of
[prop-bireduct-desc-len-formula](../propositions/bireduct-desc-len-formula.md) and
[prop-bireduct-equiv-classes-geq-bplus1](../propositions/bireduct-equiv-classes-geq-bplus1.md):

$$
BirDesc(X, B) = |X/B| \cdot (|B| + 1) \geq (|B| + 1) \cdot (|B| + 1) = (|B| + 1)^2
$$

The equality follows from the description length formula, and the inequality follows from the bound
$|X/B| \geq |B| + 1$.

## Remarks

This quadratic lower bound shows that bireducts with more attributes necessarily have longer
descriptions. The bound is tight when $|X/B| = |B| + 1$, i.e., when the bireduct induces the
minimum possible number of equivalence classes.

This result is used in the analysis of ensemble simplicity, where the total description length of
an ensemble is the sum of description lengths of its component bireducts.
