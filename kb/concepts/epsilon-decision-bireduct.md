---
id: concept-epsilon-decision-bireduct
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-06
tags: [core, bireducts, approximation]
requires:
  [concept-decision-table,
   concept-decision-bireduct,
   concept-gamma-decision-bireduct,
   concept-approximate-decision-reduct]
see_also:
  [concept-majority-function,
   concept-bireduct-ensemble,
   concept-np-hardness-foundations,
   prop-m-reduct-epsilon-bireduct-correspondence,
   prop-minimal-epsilon-bireduct-np-hard]
source: src-thesis-phd
---

# Epsilon-Decision Bireduct

An $\varepsilon$-decision bireduct is a decision bireduct that additionally satisfies a coverage
constraint: the covered object set $X$ must contain at least a $(1 - \varepsilon)$ fraction of the
universe $U$.

## Decision $\varepsilon$-Bireduct

Let $\varepsilon \in [0, 1)$ be given. A pair $(X, B)$ is a decision $\varepsilon$-bireduct if it is
a decision bireduct and:

$$
\lvert X \rvert \geq (1 - \varepsilon)\lvert U \rvert
$$

## $\gamma$-Decision $\varepsilon$-Bireduct

A pair $(X, B)$ is a $\gamma$-decision $\varepsilon$-bireduct if it is a $\gamma$-decision bireduct
satisfying the same coverage constraint:

$$
\lvert X \rvert \geq (1 - \varepsilon)\lvert U \rvert
$$

## Relationship to $\gamma$-Decision $\varepsilon$-Reducts

**Proposition.** For any $\mathbb{A}$, $(X, B)$ is a $\gamma$-decision $\varepsilon$-bireduct iff
$X = POS(B)$ and $B$ is a $\gamma$-decision $\varepsilon$-reduct. Consequently, finding a
$\gamma$-decision $\varepsilon$-bireduct with minimal $\lvert B \rvert$ is NP-hard.

## Relationship to $M$-Decision $\varepsilon$-Reducts

$B$ is the smallest $M$-decision $\varepsilon$-reduct in $\mathbb{A}$ iff there exists $X \subseteq U$
such that $(X, B)$ is a decision $\varepsilon$-bireduct and no decision $\varepsilon$-bireduct has
fewer attributes. The set $X$ is constructed as the union of objects with the most frequent decision
within each $B$-induced equivalence class. See
[prop-m-reduct-epsilon-bireduct-correspondence](../propositions/m-reduct-epsilon-bireduct-correspondence.md)
for the full statement and proof.

## Computational Complexity

**Minimal Decision $\varepsilon$-Bireduct Problem ($MD\varepsilon BP$)**: For an input $\mathbb{A}$,
find a decision $\varepsilon$-bireduct $(X, B)$ with minimal $\lvert B \rvert$.

For any $\varepsilon \in [0, 1)$, $MD\varepsilon BP$ is NP-hard. The proof reduces from the problem
of finding a minimal $M$-decision $\varepsilon$-reduct (itself NP-hard via reduction from Minimal
$\alpha$-Dominating Set). See
[prop-minimal-epsilon-bireduct-np-hard](../propositions/minimal-epsilon-bireduct-np-hard.md) for the
full statement and proof.

## Remarks

The $\varepsilon$ threshold provides a direct constraint analogous to those in [approximate decision
reducts](../concepts/approximate-decision-reduct.md) but operating explicitly on object coverage
rather than on an abstract evaluation function.

$\varepsilon$-decision bireducts are more numerous than approximative reducts for the same
$\varepsilon$, offering greater flexibility for constructing diversified
[ensembles](../concepts/bireduct-ensemble.md).
