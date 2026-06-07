---
id: prop-r-epsilon-reduct-np-hard
type: proposition
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [complexity, np-hardness, approximate-reducts, relative-gain-function]
requires:
  [concept-approximate-decision-reduct,
   concept-relative-gain-function,
   prop-m-epsilon-reduct-np-hard,
   prop-relative-r-epsilon-reduct-np-hard]
see_also:
  [prop-relative-r-epsilon-reduct-np-hard,
   prop-m-epsilon-reduct-np-hard,
   concept-approximate-decision-reduct]
source: src-thesis-phd
---

# NP-Hardness of Minimal R-Decision Epsilon-Reduct

For any $\varepsilon \in [0, 1)$, the absolute version of the minimal $R$-decision
$\varepsilon$-reduct problem is NP-hard.

## Statement

Let $\varepsilon \in [0, 1)$ be given. The problem of finding a minimal $R$-decision
$\varepsilon$-reduct for a decision table $\mathbb{A} = (U, A \cup \{d\})$ is NP-hard.

## Proof

The proof follows the same observation as
[NP-Hardness of Minimal M-Decision Epsilon-Reduct](m-epsilon-reduct-np-hard.md) but for the
function $R$ instead of $M$. The table $\mathbb{A}^{\varepsilon*}_{\mathbb{G}}$ used in the
construction is consistent, so $R(A^{\varepsilon*}_{\mathbb{G}}) = 1$, making the relative and
absolute conditions equivalent. The coincidence $R(B) = M(B)$ on this table (used in
[NP-Hardness of Minimal Relative R-Decision Epsilon-Reduct](relative-r-epsilon-reduct-np-hard.md))
completes the chain from Minimal $\alpha(\varepsilon)$-Dominating Set.
