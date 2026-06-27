---
id: prop-bireducts-0-and-1-attrs-desc-size
type: proposition
status: draft
created: 2026-06-21
updated: 2026-06-21
tags: [complexity, bireducts, ensemble]
requires:
  - prop-set-cover-construction
  - prop-solution-bireduct-properties
  - prop-bireduct-desc-len-formula
see_also:
  - prop-bireduct-replacement
  - concept-bireduct-ensemble
source: src-reduct-matrix-ensembles
---

# Description Lengths of Bireducts with 0 or 1 Attributes

In the transformed table $\mathbb{A}_{\mathcal{S}}$, the description length of a bireduct
depends solely on the cardinality of its attribute set. This lemma provides exact values for
the only two cardinalities that ultimately matter in the reduction: $0$ and $1$.

## Statement

Let $\mathbb{A}_{\mathcal{S}}$ be the decision table constructed in
[the construction](set-cover-construction.md). For a decision bireduct $(X, B)$ in
$\mathbb{A}_{\mathcal{S}}$:

- If $B = \emptyset$, then $BireductDescLen(X, \emptyset) = 1$.
- If $B = \{b\}$ for some $b \in A_{\mathcal{S}}$, then
  $BireductDescLen(X, \{b\}) = 4$.

## Proof

For $B = \emptyset$, there is a single decision rule with empty antecedent and a consequent
indicating the decision value, hence:

$$BireductDescLen(X, \emptyset) = 1.$$

For a single-attribute decision bireduct $(X, \{b\})$, if $b$ corresponded to
$S_i = \emptyset$ then every object would take value $0$ on $b$, implying
$IND(\{b\}) = IND(\emptyset)$ -- that would contradict the irreducibility of the bireduct
$(X, \{b\})$. Hence $b$ comes from a non-empty $S_i \in \mathcal{S}$. From
[the structural lemma](solution-bireduct-properties.md), $X$ contains $u_*$ with
$b(u_*) = 0$ and at least one $u_\omega$ with $b(u_\omega) = 1$. Thus,
$|X/\{b\}| = 2$, and from [the description length formula](../../bireduct-desc-len-formula.md):

$$BireductDescLen(X, \{b\}) = |X/\{b\}| \cdot (|\{b\}| + 1) = 2 \cdot (1 + 1) = 4.$$
