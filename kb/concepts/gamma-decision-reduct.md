---
id: concept-gamma-decision-reduct
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, reduction, positive-region]
requires:
  [concept-decision-table,
   concept-indiscernibility,
   concept-positive-region,
   concept-consistency]
see_also:
  [concept-decision-reduct,
   concept-approximate-decision-reduct,
   concept-gamma-decision-bireduct]
source: tmp/phd/thesis.tex
---

# Gamma-Decision Reduct

A $\gamma$-decision reduct extends the standard decision reduct to inconsistent decision tables by
requiring preservation of the positive region rather than full consistency.

## Definition

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. A subset $B \subseteq A$ is a $\gamma$-decision reduct
for $\mathbb{A}$ if and only if it is an irreducible subset of attributes such that:

$$
\gamma(B) = \gamma(A) \quad \text{or equivalently} \quad POS(B) = POS(A)
$$

## Construction via Consistent Table

A $\gamma$-decision reduct can be characterized by transforming an inconsistent table into a
consistent one. For a given $B \subseteq A$, define a modified decision attribute
$d_B^\gamma : U \to V_d \cup \{\circledast\}$ where $\circledast \notin V_d$ is a special value:

$$
d_B^\gamma(u) =
\begin{cases}
\circledast & \text{if } u \notin POS(B), \\
d(u)       & \text{otherwise}.
\end{cases}
$$

The resulting table $\mathbb{A}_B^\gamma = (U, B \cup \{d_B^\gamma\})$ is consistent.

**Proposition.** Let an inconsistent $\mathbb{A}$ be given. $B \subseteq A$ is a $\gamma$-decision
reduct in $\mathbb{A}$ iff $B$ is a decision reduct in $\mathbb{A}_A^\gamma$.

## Boolean Formula Characterization

**Proposition.** Let $\mathbb{A} = (U, A \cup \{d\})$ be given. Consider the Boolean formula:

$$
\tau^\gamma = \bigwedge_{u_i \in POS(A)}
              \bigwedge_{\substack{u_j \in U \\ d(u_i) \neq d(u_j)}}
              \bigvee_{\substack{a \in A \\ a(u_i) \neq a(u_j)}}
              \overline{a}
$$

A subset $B \subseteq A$ is a $\gamma$-decision reduct iff
$\bigwedge_{a \in B} \overline{a}$ is a prime implicant for $\tau^\gamma$.

This differs from the standard reduct formula by only considering object pairs where the first
element belongs to $POS(A)$ -- objects outside the positive region are ignored.

## Remarks

Objects in $POS(B)$ support deterministic decision rules (confidence = 1). The role of
$\gamma$-decision reducts is to use possibly small subsets of attributes to cover data with
deterministic rules as thoroughly as the full attribute set allows.

For consistent tables, $\gamma(A) = 1$ and $\gamma$-decision reducts coincide with standard decision
reducts.
