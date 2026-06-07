---
id: prop-alpha-dominating-set-np-hard
type: proposition
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [complexity, np-hardness, graph-theory]
requires:
  [concept-np-hardness-foundations]
see_also:
  [prop-relative-m-epsilon-reduct-np-hard,
   prop-minimal-dominating-set-np-hard,
   concept-np-hardness-foundations]
source: tmp/phd/thesis.tex
---

# NP-Hardness of the Minimal Alpha-Dominating Set Problem

For any $\alpha \in (0, 1]$, the Minimal $\alpha$-Dominating Set Problem extends the classical
dominating set problem with a relaxed coverage ratio and remains NP-hard.

## Statement

For any $\alpha \in (0, 1]$, the Minimal $\alpha$-Dominating Set Problem is NP-hard.

## Proof

See Appendices in Slezak, D. (2000). *Normalized Decision Functions and Measures for Inconsistent
Decision Tables Analysis*. Fundamenta Informaticae, 44(3), 291-319.

## Remarks

This result generalizes the classical Minimal Dominating Set problem (which corresponds to
$\alpha = 1$). The $\alpha$-dominating set serves as the base for the polynomial reduction to
Minimal Relative $M$-Decision $\varepsilon$-Reduct.
