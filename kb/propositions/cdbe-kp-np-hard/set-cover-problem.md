---
id: prop-set-cover-problem
type: proposition
status: draft
created: 2026-06-21
updated: 2026-06-21
tags: [complexity]
requires: []
see_also:
  - prop-set-cover-construction
  - concept-np-hardness-foundations
source: src-reduct-matrix-ensembles
---

# Set Cover Problem

The decision version of the Set Cover problem -- one of Karp's 21 NP-complete problems
(cite:karp1972reducibility) -- asks whether a family of subsets contains a bounded subfamily
whose union still covers the whole universe.

## Statement

Let $W$ be a finite universe of objects and let $\mathcal{S} = \{S_1, \dots, S_n\}$ be a finite
family of $n$ subsets of $W$ such that $\bigcup \mathcal{S} = W$. By the decision version of the
Set Cover problem we understand the task of determining whether there exists a subfamily
$\mathcal{C} \subseteq \mathcal{S}$ of size at most $l$ whose union also equals $W$:

$$\exists\; \mathcal{C} \subseteq \mathcal{S},\; |\mathcal{C}| \leq l,\; \bigcup \mathcal{C} = W.$$

An element of $W$ is denoted by $\omega$. A subfamily $\mathcal{C}$ satisfying the above
condition is called a **set cover** of $W$ of size $|\mathcal{C}|$.

## Remarks

The Set Cover problem is one of the classical NP-complete problems (Karp, 1972). In the reduction
to CDBEkP, we use the decision variant with parameter $l$ controlling the maximum allowed size of
the subfamily.
