---
id: concept-consistency
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, consistency]
requires: [concept-decision-table, concept-indiscernibility]
see_also:
  [concept-decision-reduct, concept-positive-region, concept-gamma-decision-reduct]
source: src-thesis-phd
---

# Consistency of Decision Tables

Consistency is a fundamental property of decision tables that determines whether decision reducts (in
the classical sense) exist. A table is consistent when no two objects with identical conditional
attribute values have different decisions.

## Definition

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. $\mathbb{A}$ is consistent if and only if:

$$
IND(A) \subseteq IND(\{d\})
$$

$\mathbb{A}$ is inconsistent if and only if it is not consistent.

## Equivalent Formulations

The condition $IND(A) \subseteq IND(\{d\})$ can be expressed equivalently as:

- The attribute subset $A$ discerns all objects $u_i, u_j \in U$ such that $d(u_i) \neq d(u_j)$.
- All objects $u_i, u_j$ indiscernible by $A$ have the same decision value: $d(u_i) = d(u_j)$.

In terms of the positive region, $\mathbb{A}$ is consistent iff $POS(A) = U$ (or equivalently
$\gamma(A) = 1$).

## Remarks

For consistent tables, the notions of decision reduct, $\gamma$-decision reduct, and approximate
decision reducts (with $F \in \{\gamma, M, R\}$ and $\varepsilon = 0$) all coincide.

For inconsistent tables, classical decision reducts do not exist. Extensions such as
[discernibility-based reducts](../concepts/reducts.md),
[$\gamma$-decision reducts](../concepts/gamma-decision-reduct.md), and [approximate
reducts](../concepts/approximate-decision-reduct.md) provide alternative reduction frameworks.
