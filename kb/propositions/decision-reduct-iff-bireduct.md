---
id: prop-decision-reduct-iff-bireduct
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [core, bireducts, reduction]
requires:
  [concept-decision-reduct,
   concept-decision-bireduct]
see_also:
  [prop-decision-bireduct-iff-reduct,
   concept-gamma-decision-bireduct,
   prop-monotony-properties]
source: tmp/phd/thesis.tex
---

# Decision Reduct iff Universe Bireduct

A subset $B \subseteq A$ is a decision reduct if and only if $(U, B)$ is a decision bireduct.
This embeds classical reducts into the bireduct framework.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ and $B \subseteq A$ be given. $B$ is a decision reduct if
and only if $(U, B)$ is a decision bireduct.

## Proof

**($\Rightarrow$)** If $B$ is a decision reduct, then from the definition of decision reduct,
$B$ is an irreducible subset of attributes that discerns all pairs $u_i, u_j \in U$ such that
$d(u_i) \neq d(u_j)$. Thus, $(U, B)$ satisfies all three conditions of a decision bireduct:
determination ($B \Rrightarrow_U d$), attribute irreducibility (no proper $B' \subsetneq B$
satisfies $B' \Rrightarrow_U d$), and object maximality (no proper superset of $U$ exists).

**($\Leftarrow$)** If $(U, B)$ is a decision bireduct, then from the definition of decision
bireduct we know that $B \Rrightarrow_U d$ and there is no proper $B' \subsetneq B$ such that
$B' \Rrightarrow_U d$. Thus, $B$ is an irreducible subset of attributes discerning all pairs
in $U$ with different decisions, which is exactly a decision reduct.

## Remarks

This result shows that decision reducts are a special case of decision bireducts where the
object subset equals the full universe $U$. Combined with the diagonal table transformation
(see [Decision Bireduct](../concepts/decision-bireduct.md)), it provides a bridge allowing
standard reduct algorithms to be applied to the bireduct search problem.
