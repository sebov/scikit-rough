---
id: prop-m-epsilon-reduct-np-hard
type: proposition
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [complexity, np-hardness, approximate-reducts, majority-function]
requires:
  [concept-approximate-decision-reduct,
   concept-majority-function,
   prop-relative-m-epsilon-reduct-np-hard]
see_also:
  [prop-relative-m-epsilon-reduct-np-hard,
   prop-r-epsilon-reduct-np-hard,
   concept-approximate-decision-reduct]
source: src-thesis-phd
---

# NP-Hardness of Minimal M-Decision Epsilon-Reduct

For any $\varepsilon \in [0, 1)$, the absolute version of the minimal $M$-decision
$\varepsilon$-reduct problem is NP-hard, as a direct corollary of the relative case.

## Statement

Let $\varepsilon \in [0, 1)$ be given. The problem of finding a minimal $M$-decision
$\varepsilon$-reduct for a decision table $\mathbb{A} = (U, A \cup \{d\})$ is NP-hard.

## Proof

The construction in the proof of
[NP-Hardness of Minimal Relative M-Decision Epsilon-Reduct](relative-m-epsilon-reduct-np-hard.md)
produces a consistent decision table $\mathbb{A}^{\varepsilon*}_{\mathbb{G}}$ (the transformed
table $A^{\varepsilon*}_{\mathbb{G}}$ is consistent). For consistent tables, $M(A) = 1$, so the
relative condition $M(B) \geq (1 - \varepsilon)M(A)$ is equivalent to the absolute condition
$M(B) \geq 1 - \varepsilon$. Therefore the proof of NP-hardness for $M$-decision
$\varepsilon$-reducts is identical to the proof for relative $M$-decision $\varepsilon$-reducts.
