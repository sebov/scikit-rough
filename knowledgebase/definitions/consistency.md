---
tags: [rst, core]
related: [definitions/decision_table.md, definitions/indiscernibility.md]
---
# Consistency of Decision Tables

## Consistent Decision Table

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. We say that $\mathbb{A}$ is
**consistent**, if and only if: $$IND(A) \subseteq IND(d)$$

If this condition is not met, the decision table is said to be **inconsistent**.

## Equivalent Formulations

The condition $IND(A) \subseteq IND(d)$ can be equivalently expressed in the following ways:

- The attribute subset $A$ discerns all objects $u_i, u_j \in U$ such that their decision values
  differ: $d(u_i) \neq d(u_j)$.
- All objects $u_i, u_j$ that are indiscernible by attributes from $A$ must have the same value of
  the decision attribute: $d(u_i) = d(u_j)$.
