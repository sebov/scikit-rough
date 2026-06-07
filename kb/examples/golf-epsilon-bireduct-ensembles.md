---
id: ex-golf-epsilon-bireduct-ensembles
type: example
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [example, golf, ensembles, bireducts]
requires:
  [concept-decision-table,
   concept-bireduct-ensemble,
   concept-epsilon-decision-bireduct]
see_also:
  [ex-golf-all-bireducts,
   ex-golf-epsilon-bireducts-m-reducts,
   prop-ensemble-np-hard]
source: tmp/phd/include/ensembles_decision_epsilon_bireducts.tex
---

# Golf Dataset -- Epsilon-Bireduct Ensembles

Six correct 3-element decision $\varepsilon$-bireduct ensembles for the golf dataset with
$\varepsilon = 4/14$. Each ensemble achieves majority-vote correctness (every object is covered by at
least 2 of the 3 bireducts). The coverage count shows how many bireducts in the ensemble cover each
object.

## Ensemble 1

| Component | $X$ (objects) | $B$ (attributes) |
|:----------|:--------------|:-----------------|
| 1 | $\{1,2,5,7,8,9,10,11,13,14\}$ | $\{H\}$ |
| 2 | $\{1,2,3,4,6,7,8,9,11,12,13\}$ | $\{O,H\}$ |
| 3 | $\{1,2,3,4,5,6,7,8,10,12,13,14\}$ | $\{O,W\}$ |

Coverage: objects 1-2 (3 votes), 3-6 (2), 7-8 (3), 9 (2), 10 (2), 11 (2), 12 (2), 13 (3), 14 (2).

## Ensemble 2

| Component | $X$ | $B$ |
|:----------|:----|:----|
| 1 | $\{1,2,5,7,8,9,10,11,13,14\}$ | $\{H\}$ |
| 2 | $\{1,\ldots,14\}$ | $\{O,T,W\}$ |
| 3 | $\{2,3,4,5,6,9,10,11,12,13\}$ | $\{T,W\}$ |

Coverage: objects 1 (2), 2 (3), 3 (2), 4 (2), 5 (3), 6 (2), 7 (2), 8 (2), 9 (3), 10 (3), 11 (3),
12 (2), 13 (3), 14 (2).

## Ensemble 3

| Component | $X$ | $B$ |
|:----------|:----|:----|
| 1 | $\{1,2,3,4,5,7,8,10,12,13\}$ | $\{O\}$ |
| 2 | $\{1,2,3,6,7,8,9,11,12,13,14\}$ | $\{O,H\}$ |
| 3 | $\{3,4,5,6,7,9,10,11,12,13,14\}$ | $\{O,W\}$ |

Coverage: objects 1 (2), 2 (2), 3 (3), 4 (2), 5 (2), 6 (2), 7 (3), 8 (2), 9 (2), 10 (2), 11 (2),
12 (3), 13 (3), 14 (2).

## Ensemble 4

| Component | $X$ | $B$ |
|:----------|:----|:----|
| 1 | $\{2,3,4,5,7,9,10,11,13,14\}$ | $\{H,W\}$ |
| 2 | $\{1,2,3,4,6,7,8,9,11,12,13\}$ | $\{O,H\}$ |
| 3 | $\{1,2,3,4,5,6,7,8,10,12,13,14\}$ | $\{O,W\}$ |

Coverage: objects 1 (2), 2 (3), 3 (3), 4 (3), 5 (2), 6 (2), 7 (3), 8 (2), 9 (2), 10 (2), 11 (2),
12 (2), 13 (3), 14 (2).

## Ensemble 5

| Component | $X$ | $B$ |
|:----------|:----|:----|
| 1 | $\{1,2,3,4,5,7,8,10,12,13\}$ | $\{O\}$ |
| 2 | $\{1,\ldots,14\}$ | $\{O,T,W\}$ |
| 3 | $\{2,3,5,6,8,9,10,11,13,14\}$ | $\{T,H,W\}$ |

Coverage: objects 1 (2), 2 (3), 3 (3), 4 (2), 5 (3), 6 (2), 7 (2), 8 (3), 9 (2), 10 (3), 11 (2),
12 (2), 13 (3), 14 (2).

## Ensemble 6

| Component | $X$ | $B$ |
|:----------|:----|:----|
| 1 | $\{1,2,3,5,7,8,9,12,13,14\}$ | $\{O,T\}$ |
| 2 | $\{1,3,4,5,6,7,8,10,11,12,13,14\}$ | $\{O,W\}$ |
| 3 | $\{2,3,4,5,6,9,10,11,12,13\}$ | $\{T,W\}$ |

Coverage: objects 1 (2), 2 (2), 3 (3), 4 (2), 5 (3), 6 (2), 7 (2), 8 (2), 9 (2), 10 (2), 11 (2),
12 (3), 13 (3), 14 (2).

## Remarks

An ensemble is correct if every object is covered by a strict majority of the component bireducts.
With $m=3$ bireducts, each object needs coverage count $\geq 2$. All 6 ensembles shown satisfy this
constraint.
