---
id: prop-bireduct-attrs-subset-form-bireduct
type: proposition
status: draft
created: 2026-06-17
updated: 2026-06-17
tags: [bireducts, complexity]
requires:
  - concept-decision-bireduct
see_also:
  - prop-bireduct-equiv-classes-geq-bplus1
source: src-reduct-matrix-ensembles
---

# Bireduct Attribute Subset Extension

For any decision bireduct $(X, B)$ and any subset $B' \subseteq B$, there exists a subset $X' \subseteq U$ such that $(X', B')$ is a decision bireduct.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be a decision table and $(X, B)$ be a decision bireduct for $\mathbb{A}$. For any $B' \subseteq B$, there exists $X' \subseteq U$ such that $(X', B')$ is a decision bireduct for $\mathbb{A}$.

## Proof

**Status**: INCOMPLETE

The source (`tmp/pub/main.tex:331-338`) contains a proof sketch in Polish that cuts off mid-argument. The general case remains unverified.

**Attempted approach** (from session log):
- For each equivalence class $[u]_{B'}$ of $U/B'$, we need to select objects with a single decision value
- The challenge: $U/B'$ classes are unions of $U/B$ classes, and we must ensure the resulting $X'$ is maximal
- Restricting to $X'_i \subseteq X$ is not automatic - depends on choosing the right decision for each union

**Open questions**:
- Can we always construct such $X'$ by choosing appropriate decision values?
- Or are there cases where no valid $X'$ exists?

## Remarks

This proposition is used in the induction proof of [prop-bireduct-equiv-classes-geq-bplus1](bireduct-equiv-classes-geq-bplus1.md) to establish the inductive step.

If this proposition cannot be proven in general, it may still hold in the specific context of the NP-hardness proof where the decision table has a particular structure.
