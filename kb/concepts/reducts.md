---
id: concept-decision-reduct
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, reduction]
requires:
  [concept-decision-table,
   concept-indiscernibility,
   concept-consistency,
   concept-formulae]
see_also:
  [concept-gamma-decision-reduct,
   concept-approximate-decision-reduct,
   concept-decision-bireduct,
   concept-discernibility-measure]
source: tmp/phd/thesis.tex
---

# Decision Reduct

A decision reduct is an irreducible subset of conditional attributes that preserves the ability to
discern objects with different decision values. It is the fundamental attribute reduction concept in
rough set theory.

## Decision Reduct (Classical)

Let a consistent decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. A subset $B \subseteq A$
is a decision reduct for $\mathbb{A}$ if and only if it is an irreducible subset of attributes such
that:

$$
IND(B) \subseteq IND(\{d\})
$$

Equivalently, $B$ is a decision reduct iff it is an irreducible subset of attributes that discerns
each pair $u_i, u_j \in U$ for which $d(u_i) \neq d(u_j)$.

For inconsistent decision tables, classical decision reducts do not exist.

## Discernibility-Based Decision Reduct

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. A subset $B \subseteq A$ is a discernibility-based
decision reduct for $\mathbb{A}$ if and only if it is an irreducible subset of attributes that
discerns the same pairs of objects with different decision values as $A$:

$$
\forall_{u, u' \in U}\; (u \; DIS(A) \; u' \land d(u) \neq d(u'))
  \implies (u \; DIS(B) \; u')
$$

This formulation extends the reduct concept to potentially inconsistent tables.

## Minimal Decision Reduct

A decision reduct is called minimal if its cardinality $\lvert B \rvert$ is minimal among all
decision reducts. Finding a minimal decision reduct is NP-hard.

## Boolean Formula Characterization

A general method for computing decision reducts uses Boolean formulae (Skowron & Rauszer, 1992):

**Proposition.** Let a consistent $\mathbb{A} = (U, A \cup \{d\})$ be given. Consider the Boolean
formula with propositional variables $\overline{a}$ for $a \in A$:

$$
\tau = \bigwedge_{\substack{u_i, u_j \in U \\ i < j,\; d(u_i) \neq d(u_j)}}
       \bigvee_{\substack{a \in A \\ a(u_i) \neq a(u_j)}}
       \overline{a}
$$

A subset $B \subseteq A$ is a decision reduct iff the Boolean formula
$\bigwedge_{a \in B} \overline{a}$ is a prime implicant for $\tau$.

## Remarks

A decision reduct $B \subseteq A$ determines decision values within $\mathbb{A}$: $U$ can be covered
by decision rules with predecessors built from descriptors over $B$ and successors of the form
$d = v_d$. This is equivalent to the concept of functional dependency from relational databases.

For the golf dataset, there are two decision reducts:
$\{\text{Outlook}, \text{Temperature}, \text{Wind}\}$ and
$\{\text{Outlook}, \text{Humidity}, \text{Wind}\}$.
