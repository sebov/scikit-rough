---
id: prop-gamma-epsilon-reduct-np-hard
type: proposition
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [complexity, np-hardness, reduction, approximate-reducts]
requires:
  [concept-approximate-decision-reduct,
   concept-positive-region,
   prop-relative-gamma-epsilon-reduct-np-hard]
see_also:
  [prop-relative-gamma-epsilon-reduct-np-hard,
   prop-gamma-epsilon-reduct-np-hard,
   concept-approximate-decision-reduct]
source: src-thesis-phd
---

# NP-Hardness of Minimal Gamma-Decision Epsilon-Reduct

For any $\varepsilon \in [0, 1)$, the problem of finding a minimal $\gamma$-decision
$\varepsilon$-reduct (absolute version) is NP-hard, following directly from the relative case.

## Statement

Let $\varepsilon \in [0, 1)$ be given. The problem of finding a minimal $\gamma$-decision
$\varepsilon$-reduct for a decision table $\mathbb{A} = (U, A \cup \{d\})$ is NP-hard.

## Proof

The construction of $\mathbb{A}^{\varepsilon}_{\mathbb{G}}$ in
[NP-Hardness of Minimal Relative Gamma-Decision Epsilon-Reduct](relative-gamma-epsilon-reduct-np-hard.md)
produces a consistent decision table ($\gamma(A^{\varepsilon}_{\mathbb{G}}) = 1$). For consistent
tables, the relative condition $\gamma(B) \geq (1 - \varepsilon)\gamma(A)$ is identical to the
absolute condition $\gamma(B) \geq 1 - \varepsilon$. Therefore the inequality solved to obtain the
value of $t(\varepsilon)$ has exactly the same form, and the proof for minimal $\gamma$-decision
$\varepsilon$-reducts is identical to the relative case.

Thus, if a minimal $\gamma$-decision $\varepsilon$-reduct could be found in polynomial time, the
Minimal Dominating Set problem could also be solved in polynomial time.
