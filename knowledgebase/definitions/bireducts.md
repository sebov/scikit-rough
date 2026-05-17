---
tags: [rst, core, bireducts]
related: [notation_and_symbols.md, definitions/decision_table.md, definitions/indiscernibility.md, definitions/decision_rules.md, definitions/reducts.md, definitions/approximate_reducts.md, propositions/reducts_bireducts_link.md]
---

# Decision Bireducts

Decision bireducts extend classical decision reducts by jointly selecting a subset of attributes
$B \subseteq A$ and a subset of objects $X \subseteq U$ for which that attribute subset provides a
satisfactory description of the decision. This offers a simple and flexible approach to knowledge
representation, enabling explicit analysis of how different attribute subsets cover different parts
of the data.

## Functional Dependency Within a Subset of Objects

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ and subsets $B \subseteq A$, $X \subseteq U$
be given. We say that $B$ **determines** $d$ **within** $X$, denoted $B \Rightarrow_X d$, if and
only if $B$ discerns all pairs $u_i, u_j \in X$ with different decision values:

$$
  \forall_{u_i, u_j \in X} \; d(u_i) \neq d(u_j) \implies u_i \; DIS(B) \; u_j
$$

## Monotonicity Properties

The relation $B \Rightarrow_X d$ satisfies two monotonicity properties.

**Proposition**: Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given.

1. *Monotonicity with respect to attributes*: Let $B \subseteq B' \subseteq A$ and $X \subseteq U$ be
   given. If $B \Rightarrow_X d$, then $B' \Rightarrow_X d$.
2. *Monotonicity with respect to objects*: Let $B \subseteq A$ and $X' \subseteq X \subseteq U$ be
   given. If $B \Rightarrow_X d$, then $B \Rightarrow_{X'} d$.

**Proof**:

1. Since $B \Rightarrow_X d$, for every pair $u_i, u_j \in X$ with $d(u_i) \neq d(u_j)$ there exists
   $a \in B$ such that $a(u_i) \neq a(u_j)$. Because $B \subseteq B'$, the same $a$ belongs to $B'$,
   hence $B' \Rightarrow_X d$.
2. If $B$ discerns all pairs with different decisions in $X$, then it also discerns all such pairs
   in any subset $X' \subseteq X$, hence $B \Rightarrow_{X'} d$.

## Decision Bireduct

A pair $(X, B)$ is a **decision bireduct** if and only if the following conditions hold:

1. **Determination**: $B \Rightarrow_X d$ -- the attributes $B$ determine $d$ within $X$.
2. **Attribute irreducibility**: No proper subset $B' \subsetneq B$ satisfies $B' \Rightarrow_X d$.
3. **Object maximality**: No proper superset $X' \supsetneq X$ satisfies $B \Rightarrow_{X'} d$.

Objects in $X$ are said to be **covered** by the bireduct $(X, B)$, while objects in
$U \setminus X$ are **uncovered**.

## Interpretation

A decision bireduct $(X, B)$ can be regarded as the basis for an inexact functional dependency
linking the attribute subset $B$ with the decision $d$ to the degree $X$. It was shown that $X$ is
the set-theoretic sum of objects supporting deterministic decision rules that use the values of
attributes in $B$ to describe the decision. The uncovered objects $U \setminus X$ can be treated as
outliers of the dependency $B \Rightarrow_X d$.

Every decision bireduct consists of:
- An **irreducible subset of attributes** $B$ that is sufficient to determine decisions for the
  objects in $X$.
- A **non-extendable subset of objects** $X$ for which $B$ provides a correct classification.

## Motivation

Decision bireducts were introduced to facilitate explicit analysis of whether different selected
subsets of attributes in a classifier ensemble repeat classification mistakes on the same objects
in the training data. Diversification of attribute subsets in this respect has been shown to be
important in practice, making decision bireducts a rough-set-based counterpart of ensemble methods
in machine learning.

Unlike approximate reducts, decision bireducts allow explicit analysis of whether particular reducts
repeat mistakes on the same objects, as the covered object set $X$ makes this information directly
available.
