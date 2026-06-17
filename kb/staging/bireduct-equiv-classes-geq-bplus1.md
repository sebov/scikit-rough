---
id: prop-bireduct-equiv-classes-geq-bplus1
type: proposition
status: complete
created: 2026-06-11
updated: 2026-06-11
tags: [bireducts, complexity]
requires:
  - concept-decision-bireduct
  - prop-bireduct-attrs-subset-form-bireduct
  - prop-equiv-classes-bireduct
  - prop-equiv-classes-monotonicity
  - prop-bireduct-objects-and-rules
see_also:
  - prop-bireduct-desc-len-formula
  - prop-bireduct-desc-len-geq-bplus1-squared
source: src-reduct-matrix-ensembles
---

# Bireduct Equivalence Classes Bound

For any decision bireduct, the number of equivalence classes it induces is at least $|B| + 1$.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be a decision table and $(X, B)$ be a decision bireduct for $\mathbb{A}$. Then:

$$
|X/B| \geq |B| + 1
$$

where $|X/B|$ is the number of equivalence classes determined by $IND_X(B)$ on $X$.

## Proof

We proceed by mathematical induction on $|B|$.

### Base Case: $|B| = 0$

If $B = \emptyset$, then all objects in $X$ are indiscernible (there are no attributes to distinguish them). Therefore, $|X/B| = 1 = 0 + 1 = |B| + 1$.

### Induction Step

Assume the property holds for all decision bireducts $(X, B)$ with $|B| = k \geq 0$, i.e., $|X/B| \geq k + 1$.

Let $(X', B')$ be a decision bireduct with $|B'| = k + 1$. Since $B'$ is non-empty, choose any attribute $b \in B'$ and let $B'' = B' \setminus \{b\}$, so $|B''| = k$.

By [prop-bireduct-attrs-subset-form-bireduct](bireduct-attrs-subset-form-bireduct.md), there exists a subset $X'' \subseteq U$ such that $(X'', B'')$ is a decision bireduct. By the induction hypothesis, $|X''/B''| \geq k + 1$.

By [prop-equiv-classes-bireduct](../propositions/equiv-classes-bireduct.md), we have $|X''/B''| = |U/B''|$. Therefore, $|U/B''| \geq k + 1$.

By [prop-equiv-classes-monotonicity](../propositions/equiv-classes-monotonicity.md), since $B'' \subset B'$, we have $|U/B'| \geq |U/B''|$. Therefore:

$$
|U/B'| \geq |U/B''| \geq k + 1
$$

Suppose for contradiction that $|U/B'| = k + 1$. Then $|U/B'| = |U/B''| = k + 1$, and by the second part of [prop-equiv-classes-monotonicity](../propositions/equiv-classes-monotonicity.md), we obtain $U/B' = U/B''$.

By [prop-bireduct-objects-and-rules](../propositions/bireduct-objects-and-rules.md), since $(X', B')$ is a decision bireduct, for each equivalence class $E \in U/B'$, all objects in $X' \cap E$ have the same decision value. Since $U/B' = U/B''$, the same holds for each equivalence class $E \in U/B''$: all objects in $X' \cap E$ have the same decision value.

This means $B'' \Rrightarrow_{X'} d$, where $B''$ is a proper subset of $B'$. This contradicts the irreducibility of $B'$ in the decision bireduct $(X', B')$.

Therefore, $|U/B'| \neq k + 1$, which implies $|U/B'| > k + 1$. Since cardinalities are natural numbers, $|U/B'| \geq k + 2 = (k + 1) + 1 = |B'| + 1$.

By [prop-equiv-classes-bireduct](../propositions/equiv-classes-bireduct.md) once more, $|X'/B'| = |U/B'| \geq |B'| + 1$.

### Conclusion

By mathematical induction, for any decision bireduct $(X, B)$, we have $|X/B| \geq |B| + 1$.

## Remarks

This bound provides the intuition that for a decision bireduct, the number of induced decision rules cannot drop below a specific threshold expressed in terms of the number of attributes in the bireduct. Since each equivalence class corresponds to one decision rule (by [prop-bireduct-objects-and-rules](../propositions/bireduct-objects-and-rules.md)), having at least $|B| + 1$ equivalence classes means having at least $|B| + 1$ decision rules.

This bound is tight: for example, if $B$ consists of attributes that perfectly partition $U$ into $|B| + 1$ classes, the equality holds.

Combined with [prop-bireduct-desc-len-formula](../propositions/bireduct-desc-len-formula.md), this immediately yields $BireductDescLen(X, B) \geq (|B| + 1)^2$.
