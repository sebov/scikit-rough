---
id: concept-decision-bireduct
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-06
tags: [core, bireducts, reduction]
requires:
  [concept-decision-table,
   concept-indiscernibility,
   concept-decision-reduct,
   concept-formulae]
see_also:
  [concept-gamma-decision-bireduct,
   concept-epsilon-decision-bireduct,
   concept-bireduct-ensemble,
   concept-decision-rule,
   prop-decision-reduct-iff-bireduct,
   prop-decision-bireduct-iff-reduct,
   prop-bireduct-objects-and-rules]
source: tmp/phd/thesis.tex
---

# Decision Bireduct

A decision bireduct extends the classical decision reduct by jointly selecting a subset of attributes
$B \subseteq A$ and a subset of objects $X \subseteq U$ for which those attributes provide a correct
description of the decision. It is represented as a pair $(X, B)$.

## Inexact Functional Dependency

Let $\mathbb{A} = (U, A \cup \{d\})$ and subsets $B \subseteq A$, $X \subseteq U$ be given. We say
that $B$ determines $d$ within $X$, denoted $B \Rrightarrow_X d$, if and only if $B$ discerns all
pairs $u_i, u_j \in X$ with different decision values:

$$
B \Rrightarrow_X d \;\Longleftrightarrow\;
  \forall_{u_i, u_j \in X}\; d(u_i) \neq d(u_j) \implies u_i \; DIS(B) \; u_j
$$

## Definition

A pair $(X, B)$, where $X \subseteq U$ and $B \subseteq A$, is a decision bireduct if and only if:

1. **Determination**: $B \Rrightarrow_X d$ -- the attributes $B$ determine $d$ within $X$.
2. **Attribute irreducibility**: No proper subset $B' \subsetneq B$ satisfies $B' \Rrightarrow_X d$.
3. **Object maximality**: No proper superset $X' \supsetneq X$ satisfies $B \Rrightarrow_{X'} d$.

Objects in $X$ are said to be **covered** by the bireduct; objects in $U \setminus X$ are
**uncovered**.

## Monotonicity Properties

The inexact functional dependency $B \Rrightarrow_X d$ is monotone with respect to attribute addition
($B \subseteq B' \Rightarrow B' \Rrightarrow_X d$) and object removal ($X' \subseteq X \Rightarrow B \Rrightarrow_{X'} d$).
See [prop-monotony-properties](../propositions/monotony-properties.md) for the full statement and proof.

## Relationship to Decision Reducts

$B \subseteq A$ is a decision reduct iff $(U, B)$ is a decision bireduct
([prop-decision-reduct-iff-bireduct](../propositions/decision-reduct-iff-bireduct.md)).
Conversely, $(X, B)$ is a decision bireduct iff $\mathbb{A}_X^B$ is consistent, no $X' \supsetneq X$
yields a consistent $\mathbb{A}_{X'}^B$, and $B$ is a decision reduct for $\mathbb{A}_X^B$
([prop-decision-bireduct-iff-reduct](../propositions/decision-bireduct-iff-reduct.md)).

## Interpretation via Decision Rules

Each decision bireduct $(X, B)$ induces a collection of deterministic rules whose supports sum to
exactly $X$. For each $E \in U/B$, all objects in $X \cap E$ share the same decision, and $X$ is
the union of rule supports:

$$
Rules(X, B) = \left\{
  \bigwedge_{a \in B} (a = a(u)) \Rightarrow (d = d(u)) : u \in X
\right\}
$$

See [prop-bireduct-objects-and-rules](../propositions/bireduct-objects-and-rules.md) for the full
statement and proof.

## Boolean Formula Characterization

**Proposition.** Consider the Boolean formula with propositional variables
$\overline{u_i}$ (for $i = 1, \ldots, \lvert U \rvert$) and $\overline{a}$ (for $a \in A$):

$$
\tau_{bi} = \bigwedge_{\substack{u_i, u_j \in U \\ i < j,\; d(u_i) \neq d(u_j)}}
  \left(
    \overline{u_i} \lor \overline{u_j} \lor
    \bigvee_{\substack{a \in A \\ a(u_i) \neq a(u_j)}} \overline{a}
  \right)
$$

A pair $(X, B)$ is a decision bireduct iff
$\bigwedge_{u_i \in U \setminus X} \overline{u_i} \land \bigwedge_{a \in B} \overline{a}$
is a prime implicant for $\tau_{bi}$.

## Diagonal Table Transformation

**Proposition.** For $\mathbb{A} = (U, A \cup \{d\})$, construct $\mathbb{A}^{\boxbslash}$ by adding
$\lvert U \rvert$ diagonal attributes $a_i^{\boxbslash}$ where $a_i^{\boxbslash}(u_j) = 1$ iff
$i = j$ and $0$ otherwise. Then $(X, B)$ is a decision bireduct for $\mathbb{A}$ iff
$B \cup \{a_i^{\boxbslash} : u_i \notin X\}$ is a decision reduct for $\mathbb{A}^{\boxbslash}$.

This allows applying standard reduct algorithms to the bireduct search problem.
