---
id: concept-cdbe-kp
type: concept
status: complete
created: 2026-06-20
updated: 2026-06-20
tags: [ensemble, bireducts, complexity]
requires:
  - concept-decision-table
  - concept-bireduct-ensemble
  - concept-consistency
see_also:
  - prop-correct-ensemble-iff-dectab-consistent
  - prop-ensemble-np-hard
source: src-reduct-matrix-ensembles
---

# Correct Decision Bireduct Ensemble of Size k Problem (CDBEkP)

CDBEkP is the decision problem of determining whether a decision table admits a correct ensemble of
bireducts whose total description length does not exceed a given bound $k$.

## Definition

Let $\mathbb{A} = (U, A \cup \{d\})$ be a decision table and $k \geq 0$ be an integer. The **Correct
Decision Bireduct Ensemble of Size $k$ Problem** (CDBEkP) asks:

Does there exist a correct ensemble of decision bireducts
$\mathcal{B} = \{(X_1, B_1), \ldots, (X_m, B_m)\}$ for $\mathbb{A}$ such that

$$
EnsembleDescLen(\mathcal{B}) \leq k \; ?
$$

If the answer is YES, the ensemble $\mathcal{B}$ is called a **witness**.

## Intuition

CDBEkP captures the question of whether we can find a correct ensemble that is "compact enough" --
its total description length, measured as the sum of descriptors across all induced decision rules,
does not exceed $k$. This is the decision-problem counterpart of the optimization problem SCDBEP,
which asks for a correct ensemble with the absolute minimum description length.

## Instance

An instance of CDBEkP is a pair $(\mathbb{A}, k)$ where $\mathbb{A}$ is a decision table and $k$ is
a non-negative integer. The instance is a **YES-instance** if and only if a correct ensemble
$\mathcal{B}$ exists with $EnsembleDescLen(\mathcal{B}) \leq k$.

## Remarks

If $\mathbb{A}$ is inconsistent, the answer is always NO (by
[prop-correct-ensemble-iff-dectab-consistent](../propositions/correct-ensemble-iff-dectab-consistent.md)),
regardless of $k$.

CDBEkP is NP-complete. The proof reduces from the Set Cover problem by constructing a consistent
decision table $\mathbb{A}_{\mathcal{S}}$ from a Set Cover instance $(W, \mathcal{S})$, and showing
that a correct ensemble with description length at most $k$ exists for $\mathbb{A}_{\mathcal{S}}$ if
and only if a set cover of size at most $k'$ exists for $(W, \mathcal{S})$.
