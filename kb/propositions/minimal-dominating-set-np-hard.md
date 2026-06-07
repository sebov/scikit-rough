---
id: prop-minimal-dominating-set-np-hard
type: proposition
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [complexity, np-hardness, graph-theory]
requires:
  [concept-np-hardness-foundations]
see_also:
  [prop-relative-gamma-epsilon-reduct-np-hard,
   prop-relative-m-epsilon-reduct-np-hard,
   concept-np-hardness-foundations]
source: src-thesis-phd
---

# NP-Hardness of the Minimal Dominating Set Problem

The Minimal Dominating Set Problem is one of the classical NP-hard optimization problems (Garey &
Johnson, 1979).

## Statement

The Minimal Dominating Set Problem is NP-hard.

## Proof

This is one of the most basic NP-hard problems, cf. Garey, M.R. and Johnson, D.S. (1979).
*Computers and Intractability: A Guide to the Theory of NP-Completeness*. W.H. Freeman and Company.

## Remarks

The Minimal Dominating Set problem serves as the base for one branch of the NP-hardness reduction
chain for approximate reduct problems:

- MDS $\to$ Minimal Relative $\gamma$-Decision $\varepsilon$-Reduct $\to$ Minimal $\gamma$-Decision
  $\varepsilon$-Reduct.

A parallel branch starts from the Minimal $\alpha$-Dominating Set problem (a generalization with
relaxed coverage, independently proved NP-hard in Slezak 2000):

- Minimal $\alpha$-DS $\to$ Minimal Relative $M$-Decision $\varepsilon$-Reduct $\to$ Minimal
  $M$-Decision $\varepsilon$-Reduct $\to$ Minimal Relative $R$-Decision $\varepsilon$-Reduct $\to$
  Minimal $R$-Decision $\varepsilon$-Reduct
  (the $R$ steps follow via the observation $R(B) = M(B)$ on the constructed table).
