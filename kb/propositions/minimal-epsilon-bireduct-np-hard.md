---
id: prop-minimal-epsilon-bireduct-np-hard
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [complexity, bireducts, approximate-reducts]
requires:
  [concept-epsilon-decision-bireduct,
   concept-approximate-decision-reduct,
   concept-np-hardness-foundations]
see_also:
  [prop-m-reduct-epsilon-bireduct-correspondence,
   prop-minimal-m-reduct-np-hard,
   concept-epsilon-decision-bireduct]
source: src-thesis-phd
---

# NP-Hardness of Minimal Epsilon-Bireduct Problem

For any $\varepsilon \in [0, 1)$, the problem of finding an $\varepsilon$-decision bireduct with the
minimum number of attributes is NP-hard.

## Statement

For any $\varepsilon \in [0, 1)$, the Minimal Decision $\varepsilon$-Bireduct Problem ($MD\varepsilon BP$)
is NP-hard.

## Proof

The problem of finding a minimal $M$-reduct for an input decision table is NP-hard for each
$\varepsilon \in [0, 1)$ treated as a constant (see [NP-Hardness Foundations](../concepts/np-hardness-foundations.md)).

We can propose a straightforward reduction where decision bireducts which are solutions for the
considered optimization problem yield the smallest $M$-reducts for the same data sets. Namely,
suppose that a pair $(X, B)$ is an $\varepsilon$-decision bireduct with the lowest cardinality of
$B$ for a given table $\mathbb{A} = (U, A \cup \{d\})$. Then, according to the correspondence
between $M$-reducts and $\varepsilon$-bireducts (see
[Correspondence Between M-Reducts and Epsilon-Bireducts](m-reduct-epsilon-bireduct-correspondence.md)),
the same $B$ needs to be the smallest $M$-reduct for the same $\mathbb{A}$.

Therefore, if we could solve $MD\varepsilon BP$ in polynomial time, we could also solve the minimal
$M$-reduct problem in polynomial time, which contradicts the NP-hardness of the latter.

## Remarks

This result extends the NP-hardness from approximate reducts to epsilon-bireducts. The reduction is
direct: the attribute subset $B$ in the optimal $\varepsilon$-bireduct is exactly the optimal
$M$-reduct.

The practical implication is that heuristic search methods are necessary for finding small
$\varepsilon$-bireducts in large datasets. The ordering-based and sampling algorithms discussed in
the thesis provide practical approaches, though they do not guarantee optimality.

Note that while minimizing $\lvert B \rvert$ is NP-hard, in practice it may not always be optimal to
maximize $\lvert X \rvert$. Some collections of $\varepsilon$-bireducts with relatively lower
$\lvert X \rvert$ -- but still satisfying the coverage constraint $\lvert X \rvert \geq (1 - \varepsilon)\lvert U \rvert$
-- may better "cooperate" with each other in ensemble settings.
