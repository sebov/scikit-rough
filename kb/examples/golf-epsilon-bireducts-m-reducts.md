---
id: ex-golf-epsilon-bireducts-m-reducts
type: example
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [example, golf, bireducts, approximate-reducts]
requires:
  [concept-decision-table,
   concept-epsilon-decision-bireduct,
   concept-approximate-decision-reduct,
   concept-majority-function]
see_also:
  [ex-golf-all-bireducts,
   ex-golf-bireduct-rules,
   prop-relative-m-epsilon-reduct-np-hard]
source: tmp/phd/include/m_decision_epsilon_reducts_decision_epsilon_bireducts_all.tex
---

# Golf Dataset -- Epsilon-Bireducts and M-Reducts

For the golf dataset with $\varepsilon = 4/14$, the table lists whether each attribute subset is a
minimal $M$-decision $\varepsilon$-reduct, and if so, all object subsets $X$ for which $(X, B)$ is an
$\varepsilon$-decision bireduct. If not a reduct, the bireduct objects are `NA` (no bireduct with
this $B$ exists under the $\varepsilon$ threshold).

| $B \subseteq A$ | minimal $M$-$\varepsilon$-reduct? | $X$ for $\varepsilon$-bireducts $(X, B)$ |
|:---|:---:|:---|
| $\emptyset$ | no | NA |
| $\{O\}$ | yes | $\{1,2,3,4,5,7,8,10,12,13\}$ |
| $\{T\}$ | no | NA |
| $\{H\}$ | yes | $\{1,2,5,7,8,9,10,11,13,14\}$ |
| $\{W\}$ | no | NA |
| $\{O,T\}$ | no | $\{1,2,3,4,5,7,8,9,10,12,13\}$, $\{1,2,3,4,5,7,9,10,11,12,13\}$, $\{1,2,3,4,6,7,8,9,10,12,13\}$, $\{1,2,3,4,6,7,9,10,11,12,13\}$, $\{1,2,3,5,7,8,9,12,13,14\}$, $\{1,2,3,5,7,9,11,12,13,14\}$, $\{1,2,3,6,7,8,9,12,13,14\}$, $\{1,2,3,6,7,9,11,12,13,14\}$ |
| $\{O,H\}$ | no | $\{1,2,3,4,5,7,8,9,10,11,12,13\}$, $\{1,2,3,4,6,7,8,9,11,12,13\}$, $\{1,2,3,5,7,8,9,10,11,12,13,14\}$, $\{1,2,3,6,7,8,9,11,12,13,14\}$ |
| $\{O,W\}$ | no | $\{1,2,3,4,5,6,7,8,10,12,13,14\}$, $\{1,3,4,5,6,7,8,10,11,12,13,14\}$, $\{2,3,4,5,6,7,9,10,12,13,14\}$, $\{3,4,5,6,7,9,10,11,12,13,14\}$ |
| $\{T,H\}$ | no | $\{1,2,4,5,7,9,10,11,12,13\}$ |
| $\{T,W\}$ | yes | $\{2,3,4,5,6,9,10,11,12,13\}$, $\{2,3,4,5,7,9,10,11,12,13\}$ |
| $\{H,W\}$ | no | $\{2,3,4,5,7,9,10,11,13,14\}$ |
| $\{O,T,H\}$ | no | $\{1,2,3,4,6,7,8,9,10,11,12,13\}$, $\{1,2,3,6,7,8,9,10,11,12,13,14\}$ |
| $\{O,T,W\}$ | no | $\{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$ |
| $\{O,H,W\}$ | no | $\{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$ |
| $\{T,H,W\}$ | no | $\{1,2,4,5,6,9,10,11,12,13\}$, $\{1,2,4,5,6,9,10,11,13,14\}$, $\{1,2,4,5,7,9,10,11,13,14\}$, $\{1,2,5,6,8,9,10,11,12,13\}$, $\{1,2,5,6,8,9,10,11,13,14\}$, $\{1,2,5,7,8,9,10,11,12,13\}$, $\{2,3,4,5,6,9,10,11,13,14\}$, $\{2,3,5,6,8,9,10,11,12,13\}$, $\{2,3,5,6,8,9,10,11,13,14\}$, $\{2,3,5,7,8,9,10,11,12,13\}$, $\{2,3,5,7,8,9,10,11,13,14\}$ |
| $\{O,T,H,W\}$ | no | NA |

## Remarks

Three subsets are minimal $M$-decision ($4/14$)-reducts: $\{O\}$, $\{H\}$, and $\{T,W\}$. Each such
reduct yields one or more $\varepsilon$-decision bireducts -- the corresponding $X$ sets are listed
in the third column. For non-reduct supersets, multiple bireducts may exist (e.g., $\{O,T,H\}$
yields 2), but only the listed $B$ give the attribute sets appearing in any
$\varepsilon$-bireduct.

The case $\varepsilon = 4/14 \approx 0.286$ corresponds to allowing up to 4 objects to be excluded
from the bireduct's coverage.
