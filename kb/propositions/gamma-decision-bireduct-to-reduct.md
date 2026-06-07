---
id: prop-gamma-decision-bireduct-to-reduct
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [core, bireducts, reduction, positive-region]
requires:
  [concept-decision-reduct,
   concept-gamma-decision-bireduct]
see_also:
  [prop-decision-reduct-iff-bireduct,
   concept-gamma-decision-reduct,
   prop-gamma-decision-bireduct-pos]
source: src-thesis-phd
---

# Decision Reduct iff Universe Gamma-Bireduct

A subset $B \subseteq A$ is a decision reduct if and only if $(U, B)$ is a $\gamma$-decision
bireduct. This embeds classical reducts into the $\gamma$-bireduct framework, analogous to the
standard bireduct case.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ and $B \subseteq A$ be given. $B$ is a decision reduct if
and only if $(U, B)$ is a $\gamma$-decision bireduct.

## Proof

**($\Rightarrow$)** If $B$ is a decision reduct, then from the definition of decision reduct,
$B$ is an irreducible subset of attributes that discerns all pairs $u_i, u_j \in U$ such that
$d(u_i) \neq d(u_j)$. As a result, $(U, B)$ satisfies all three conditions of a $\gamma$-decision
bireduct: gamma-determination ($B \Rrightarrow^{\gamma}_U d$, since discernibility within $U$
against $U$ is the same as discernibility within $U$), attribute irreducibility (no proper
$B' \subsetneq B$ satisfies $B' \Rrightarrow^{\gamma}_U d$), and object maximality (no proper
superset of $U$ exists).

**($\Leftarrow$)** If $(U, B)$ is a $\gamma$-decision bireduct, then from the definition of
$\gamma$-decision bireduct we know that $B \Rrightarrow^{\gamma}_U d$ and there is no $B' \subsetneq B$
such that $B' \Rrightarrow^{\gamma}_U d$. When $X = U$, the gamma functional dependency
$B \Rrightarrow^{\gamma}_U d$ is equivalent to $B \Rrightarrow_U d$ (both require discerning all
pairs in $U$ with different decisions). It means that $B$ is a decision reduct.

## Remarks

This result shows that decision reducts are a special case of $\gamma$-decision bireducts where the
object subset equals the full universe $U$. Combined with the gamma-modified table construction
(see [Gamma-Decision Reduct](../concepts/gamma-decision-reduct.md)), it provides a bridge allowing
standard reduct algorithms on $\mathbb{A}_A^\gamma$ to be applied to the $\gamma$-bireduct search
problem.
