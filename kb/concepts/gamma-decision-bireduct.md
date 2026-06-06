---
id: concept-gamma-decision-bireduct
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-06
tags: [core, bireducts, positive-region]
requires:
  [concept-decision-table,
   concept-indiscernibility,
   concept-positive-region,
   concept-decision-bireduct]
see_also:
  [concept-decision-bireduct,
   concept-gamma-decision-reduct,
   concept-epsilon-decision-bireduct,
   prop-gamma-monotony-properties,
   prop-gamma-decision-bireduct-to-reduct,
   prop-gamma-decision-bireduct-pos,
   prop-gamma-decision-bireduct-boolean-formula]
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

## Monotonicity Properties

The gamma functional dependency $B \Rrightarrow^{\gamma}_X d$ is monotone with respect to attribute
addition ($B \subseteq B' \Rightarrow B' \Rrightarrow^{\gamma}_X d$) and object removal
($X' \subseteq X \Rightarrow B \Rrightarrow^{\gamma}_{X'} d$). See
[prop-gamma-monotony-properties](../propositions/gamma-monotony-properties.md) for the full
statement and proof.

## Relationship to Positive Region

$(X, B)$ is a $\gamma$-decision bireduct iff $X = POS(B)$ and there is no proper subset
$B' \subsetneq B$ with $POS(B') = POS(B)$. This means the covered object set in a $\gamma$-decision
bireduct is always exactly the positive region of its attribute subset. For a given $B$, there can
be at most one $\gamma$-decision bireduct. See
[prop-gamma-decision-bireduct-pos](../propositions/gamma-decision-bireduct-pos.md) for the full
statement and proof.

## Relationship to Decision Reducts

$B$ is a decision reduct iff $(U, B)$ is a $\gamma$-decision bireduct. See
[prop-gamma-decision-bireduct-to-reduct](../propositions/gamma-decision-bireduct-to-reduct.md) for
the full statement and proof.

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

$\gamma$-decision bireducts correspond to prime implicants of a Boolean formula $\tau_{bi}^{\gamma}$
that is more restrictive than the standard bireduct formula. Consider the formula:

$$
\tau_{bi}^{\gamma} = \bigwedge_{u_i \in U}
  \bigwedge_{\substack{u_j \in U \\ d(u_i) \neq d(u_j)}}
  \left(
    \overline{i} \lor
    \bigvee_{\substack{a \in A \\ a(u_i) \neq a(u_j)}} \overline{a}
  \right)
$$

A pair $(X, B)$ is a $\gamma$-decision bireduct iff
$\bigwedge_{u_i \in U \setminus X} \overline{i} \land \bigwedge_{a \in B} \overline{a}$
is a prime implicant for $\tau_{bi}^{\gamma}$. See
[prop-gamma-decision-bireduct-boolean-formula](../propositions/gamma-decision-bireduct-boolean-formula.md)
for the full statement and proof.
