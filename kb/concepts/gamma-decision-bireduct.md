---
id: concept-gamma-decision-bireduct
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, bireducts, positive-region]
requires:
  [concept-decision-table,
   concept-indiscernibility,
   concept-positive-region,
   concept-decision-bireduct]
see_also:
  [concept-decision-bireduct,
   concept-gamma-decision-reduct,
   concept-epsilon-decision-bireduct]
source: tmp/phd/thesis.tex
---

# Gamma-Decision Bireduct

A $\gamma$-decision bireduct is a variant of the decision bireduct where the functional dependency
requires discernibility against the entire universe $U$, not just within the covered object subset.

## Gamma Functional Dependency

Let $\mathbb{A} = (U, A \cup \{d\})$ and subsets $X \subseteq U$, $B \subseteq A$ be given. We say
that $B$ $\gamma$-determines $d$ within $X$, denoted $B \Rrightarrow^{\gamma}_X d$, if and only if
$B$ discerns all pairs $u_i \in X$, $u_j \in U$ with different decision values:

$$
B \Rrightarrow^{\gamma}_X d \;\Longleftrightarrow\;
  \forall_{u_i \in X}\; \forall_{u_j \in U}\;
  d(u_i) \neq d(u_j) \implies u_i \; DIS(B) \; u_j
$$

## Definition

A pair $(X, B)$ is a $\gamma$-decision bireduct if and only if:

1. **Determination**: $B \Rrightarrow^{\gamma}_X d$.
2. **Attribute irreducibility**: No proper $B' \subsetneq B$ satisfies $B' \Rrightarrow^{\gamma}_X d$.
3. **Object maximality**: No proper $X' \supsetneq X$ satisfies $B \Rrightarrow^{\gamma}_{X'} d$.

## Relationship to Positive Region

**Proposition.** $(X, B)$ is a $\gamma$-decision bireduct iff:

1. $X = POS(B)$, and
2. There is no proper subset $B' \subsetneq B$ with $POS(B') = POS(B)$.

This means the covered object set in a $\gamma$-decision bireduct is always exactly the positive
region of its attribute subset. For a given $B$, there can be at most one $\gamma$-decision
bireduct.

## Relationship to Decision Reducts

**Proposition.** $B$ is a decision reduct iff $(U, B)$ is a $\gamma$-decision bireduct.

$\gamma$-decision bireducts can be transformed to the problem of searching decision reducts in the
modified consistent table $\mathbb{A}_A^\gamma$.

## Comparison with Decision Bireducts

$\gamma$-decision bireducts impose a stronger condition: an object can belong to $X$ only if it is
discerned from all objects with different decisions in the **entire** $U$, not just within $X$.
Consequently:

- For the same $B$, $X$ is usually smaller than in a standard decision bireduct.
- Decision rules generated from $\gamma$-decision bireducts are deterministic with respect to the
  whole $U$ (confidence = 1), not just within $X$.
- There are generally fewer $\gamma$-decision bireducts than decision bireducts.

## Boolean Formula Characterization

**Proposition.** Consider the Boolean formula:

$$
\tau_{bi}^{\gamma} = \bigwedge_{u_i \in U}
  \bigwedge_{\substack{u_j \in U \\ d(u_i) \neq d(u_j)}}
  \left(
    \overline{u_i} \lor
    \bigvee_{\substack{a \in A \\ a(u_i) \neq a(u_j)}} \overline{a}
  \right)
$$

A pair $(X, B)$ is a $\gamma$-decision bireduct iff
$\bigwedge_{u_i \in U \setminus X} \overline{u_i} \land \bigwedge_{a \in B} \overline{a}$
is a prime implicant for $\tau_{bi}^{\gamma}$.
