---
id: prop-bireduct-desc-len-formula
type: proposition
status: complete
created: 2026-06-11
updated: 2026-06-11
tags: [bireducts, complexity, ensemble]
requires:
  - concept-bireduct-ensemble
  - concept-decision-rule
  - prop-equiv-classes-bireduct
see_also: []
source: src-reduct-matrix-ensembles
---

# Bireduct Description Length Formula

The description length of a decision bireduct equals the number of equivalence classes it induces, multiplied by the number of descriptors per rule.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be a decision table and $(X, B)$ be a decision bireduct for $\mathbb{A}$. The description length of $(X, B)$ is:

$$
BirDesc(X, B) = |X/B| \cdot (|B| + 1)
$$

where $|X/B|$ is the number of equivalence classes determined by $IND_X(B)$ on $X$.

## Proof

From the definition of $BirDesc$, the description length is the total number of descriptors used in the set of decision rules induced from the bireduct:

$$
Rules(X, B) = \left\{ \bigwedge_{a \in B} (a = a(u)) \Rightarrow (d = d(u)) : u \in X \right\}
$$

Each decision rule consists of:
- An antecedent: a conjunction of $|B|$ descriptors, one for each attribute in $B$
- A consequent: a single descriptor for the decision attribute $d$

Therefore, each decision rule contains exactly $|B| + 1$ descriptors.

Two objects $u_1, u_2 \in X$ generate the same decision rule if and only if they have the same values on all attributes in $B$. This is equivalent to saying $u_1$ and $u_2$ belong to the same equivalence class in $X/B$. Hence, the number of distinct rules in $Rules(X, B)$ equals $|X/B|$.

The total number of descriptors is the number of distinct rules multiplied by the number of descriptors per rule:

$$
BirDesc(X, B) = |X/B| \cdot (|B| + 1)
$$

## Remarks

By [prop-equiv-classes-bireduct](equiv-classes-bireduct.md), we know that $|X/B| = |U/B|$ for any decision bireduct $(X, B)$. Therefore, the formula can equivalently be written as:

$$
BirDesc(X, B) = |U/B| \cdot (|B| + 1)
$$

This alternative form connects the bireduct's description length directly to the global partition structure of $U$ under $IND(B)$.
