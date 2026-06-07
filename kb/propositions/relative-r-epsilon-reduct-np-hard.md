---
id: prop-relative-r-epsilon-reduct-np-hard
type: proposition
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [complexity, np-hardness, approximate-reducts, relative-gain-function]
requires:
  [concept-approximate-decision-reduct,
   concept-relative-gain-function,
   prop-relative-m-epsilon-reduct-np-hard]
see_also:
  [prop-relative-m-epsilon-reduct-np-hard,
   prop-r-epsilon-reduct-np-hard,
   concept-approximate-decision-reduct,
   concept-relative-gain-function]
source: src-thesis-phd
---

# NP-Hardness of Minimal Relative R-Decision Epsilon-Reduct

For any $\varepsilon \in [0, 1)$, the minimal relative $R$-decision $\varepsilon$-reduct problem is
NP-hard, following from the $M$ case with the observation that $R(B) = M(B)$ on the constructed
table.

## Statement

Let $\varepsilon \in [0, 1)$ be given. The problem of finding a minimal relative $R$-decision
$\varepsilon$-reduct for a decision table $\mathbb{A} = (U, A \cup \{d\})$ is NP-hard.

## Proof

Reuse the construction and calculations from
[NP-Hardness of Minimal Relative M-Decision Epsilon-Reduct](relative-m-epsilon-reduct-np-hard.md).
The decision table $\mathbb{A}^{\varepsilon*}_{\mathbb{G}}$ constructed there remains the input.
It suffices to observe that on this specific table, the relative gain function $R$ coincides with
the majority function $M$: for any $B \subseteq A^{\varepsilon*}_{\mathbb{G}}$,

$$
R(B) = M(B).
$$

Therefore the relative $R$-decision $\varepsilon$-superreduct condition $R(B) \geq
(1 - \varepsilon)R(A^{\varepsilon*}_{\mathbb{G}})$ is identical to the relative $M$-decision
$\varepsilon$-superreduct condition. All steps from the $M$ proof carry over unchanged, and the
reduction from Minimal $\alpha(\varepsilon)$-Dominating Set remains valid.

Thus the problem of finding a minimal relative $R$-decision $\varepsilon$-reduct is NP-hard.
