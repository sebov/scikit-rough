---
id: prop-decision-bireduct-iff-reduct
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [core, bireducts, reduction]
requires:
  [concept-decision-bireduct,
   concept-decision-reduct,
   concept-consistency]
see_also:
  [prop-decision-reduct-iff-bireduct,
   concept-decision-bireduct,
   concept-gamma-decision-bireduct]
source: src-thesis-phd
---

# Decision Bireduct via Subtable Reduct

A decision bireduct $(X, B)$ can be characterized in terms of standard decision reducts applied
to the subtable $\mathbb{A}_X^B$ obtained by restricting objects to $X$ and attributes to $B$.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. For $B \subseteq A$ and $X \subseteq U$, denote
by $\mathbb{A}_X^B$ the decision table obtained from $\mathbb{A}$ by removing objects outside
$X$ and attributes outside $B$. Then $(X, B)$ is a decision bireduct for $\mathbb{A}$ if and
only if both of the following conditions hold:

1. $\mathbb{A}_X^B$ is consistent and there is no $X' \supsetneq X$ such that $\mathbb{A}_{X'}^B$
   is consistent.
2. $B$ is a decision reduct for $\mathbb{A}_X^B$.

## Proof

**($\Rightarrow$)** If $(X, B)$ is a decision bireduct, then from the definition of decision
bireduct we have that $B \Rrightarrow_X d$ and there is no proper subset $B' \subsetneq B$ such
that $B' \Rrightarrow_X d$. Thus, $B$ is a decision reduct for $\mathbb{A}_X^B$. This implies
also that $\mathbb{A}_X^B$ is consistent. Finally, the fact that there is no proper superset
$X' \supsetneq X$ such that $B \Rrightarrow_{X'} d$ implies that there is no $X' \supsetneq X$
such that $\mathbb{A}_{X'}^B$ is consistent.

**($\Leftarrow$)** We need to check the three properties that define a decision bireduct. If $B$
is a decision reduct for $\mathbb{A}_X^B$, then from the definition of decision reduct we know
that $B$ is an irreducible subset of attributes that discerns all pairs $u_i, u_j \in X$ such
that $d(u_i) \neq d(u_j)$. Thus, $B \Rrightarrow_X d$ and there is no proper subset $B'
\subsetneq B$ such that $B' \Rrightarrow_X d$. Finally, using the statement that there is no
$X' \supsetneq X$ such that $\mathbb{A}_{X'}^B$ is consistent, we have the third property of the
decision bireduct definition, i.e., there is no proper superset $X' \supsetneq X$ such that
$B \Rrightarrow_{X'} d$.

## Remarks

This result helps adapt algorithms developed for searching standard decision reducts to the
bireduct setting. Given a candidate object set $X$, one checks consistency of $\mathbb{A}_X^B$
and then computes a standard reduct within that subtable. The maximality condition on $X$
ensures that the object subset cannot be enlarged without introducing inconsistency.
