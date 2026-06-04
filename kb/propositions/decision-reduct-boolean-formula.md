---
id: prop-decision-reduct-boolean-formula
type: proposition
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, reduction]
requires:
  [concept-decision-table,
   concept-indiscernibility,
   concept-consistency,
   concept-decision-reduct]
see_also:
  [prop-gamma-decision-reduct-consistent-table,
   prop-gamma-decision-reduct-boolean-formula,
   prop-decision-bireduct-boolean-formula]
source: tmp/phd/thesis.tex
---

# Decision Reduct Boolean Formula Characterisation

Decision reducts for a consistent decision table correspond exactly to prime implicants of a Boolean
formula that encodes the discernibility matrix. This provides a general computational method for
finding all decision reducts.

## Statement

Let a consistent decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. Consider the following
Boolean formula with propositional variables $\overline{a}$ for $a \in A$:

$$
\tau = \bigwedge_{\substack{u_i, u_j \in U \\ i < j,\; d(u_i) \neq d(u_j)}}
       \bigvee_{\substack{a \in A \\ a(u_i) \neq a(u_j)}}
       \overline{a}
$$

An arbitrary subset $B \subseteq A$ is a decision reduct if and only if the Boolean formula
$\bigwedge_{a \in B} \overline{a}$ is a prime implicant for $\tau$.

## Proof

See Skowron and Rauszer (1992), "The Discernibility Matrices and Functions in Information Systems".

The proof constructs a Boolean formula $\tau$ whose clauses correspond to pairs of objects from
different decision classes. Each clause is a disjunction of propositional variables representing
the attributes that discern that pair. A product of literals $\bigwedge_{a \in B} \overline{a}$
is an implicant of $\tau$ precisely when $B$ discerns every such pair -- which is the defining
condition of a decision reduct. The prime implicant condition ensures irreducibility: removing any
attribute from $B$ breaks the implicant, meaning that attribute is essential for discernibility.

## Consequences

This result reduces the problem of computing decision reducts to the well-studied problem of finding
prime implicants of a Boolean formula in conjunctive normal form. It also establishes an explicit
connection between rough set attribute reduction and Boolean reasoning, enabling the adaptation of
algorithms from Boolean function theory to the search for decision reducts.
