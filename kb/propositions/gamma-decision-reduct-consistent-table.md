---
id: prop-gamma-decision-reduct-consistent-table
type: proposition
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, reduction, positive-region]
requires:
  [concept-decision-table,
   concept-decision-reduct,
   concept-gamma-decision-reduct,
   concept-positive-region]
see_also:
  [prop-decision-reduct-boolean-formula,
   prop-gamma-decision-reduct-boolean-formula]
source: tmp/phd/thesis.tex
---

# Gamma-Decision Reduct via Consistent Table Construction

A $\gamma$-decision reduct in an inconsistent table is exactly a standard decision reduct in the
modified consistent table obtained by replacing conflicting decisions with the special value
$\circledast$.

## Statement

Let an inconsistent decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. An arbitrary subset
$B \subseteq A$ is a $\gamma$-decision reduct in $\mathbb{A}$ if and only if $B$ is a decision
reduct in $\mathbb{A}_A^\gamma = (U, A \cup \{d_A^\gamma\})$, where $d_A^\gamma$ is defined as:

$$
d_A^\gamma(u) =
\begin{cases}
\circledast & \text{if } u \notin POS(A), \\
d(u)        & \text{otherwise},
\end{cases}
$$

with $\circledast \notin V_d$ being a special decision value.

## Proof

See Slezak and Dutta (2018), "Dynamic and Discernibility Characteristics of Different Attribute
Reduction Criteria".

The transformation replaces the original decision $d$ with the modified decision $d_A^\gamma$ that
assigns the special value $\circledast$ to all objects outside the positive region $POS(A)$. Since
$\mathbb{A}_A^\gamma$ is consistent by construction (all conflicting indiscernible objects now share
the decision $\circledast$), standard decision reducts exist for it.

A subset $B$ is a decision reduct for $\mathbb{A}_A^\gamma$ precisely when $B$ is an irreducible
subset of attributes that discerns all pairs of objects in $\mathbb{A}_A^\gamma$ with different
decisions under $d_A^\gamma$. Since objects outside $POS(A)$ all carry the same decision
$\circledast$, the only relevant discernibility constraints involve objects in $POS(A)$ -- which is
exactly the condition defining a $\gamma$-decision reduct for the original table $\mathbb{A}$.

The construction extends to arbitrary $B \subseteq A$ via $\mathbb{A}_B^\gamma$, enabling
algorithmic approaches that search for decision reducts in consistently transformed tables using
standard reduct computation methods.

## Consequences

This result bridges the gap between decision reducts (which require consistency) and
$\gamma$-decision reducts (which handle inconsistency via the positive region). It allows any
algorithm for computing standard decision reducts to be applied to the problem of finding
$\gamma$-decision reducts, after an appropriate preprocessing step that constructs the modified
consistent table.
