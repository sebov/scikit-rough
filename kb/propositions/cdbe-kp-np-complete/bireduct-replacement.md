---
id: prop-bireduct-replacement
type: proposition
status: complete
created: 2026-07-07
updated: 2026-07-07
tags: [complexity, bireducts, ensemble]
requires:
  - prop-set-cover-construction
  - prop-solution-bireduct-properties
  - prop-bireducts-0-and-1-attrs-desc-size
  - prop-bireduct-desc-len-geq-bplus1-squared
  - concept-bireduct-ensemble
see_also:
  - prop-cdbe-kp-np-complete
source: "Slezak & Stawicki, 'Complexity of Searching for the Simplest Reduct Matrix Ensembles'
  (paper in preparation)"
---

# Bireduct Replacement: Correctness and Simplicity

In the transformed table $\mathbb{A}_{\mathcal{S}}$, any correct ensemble containing a bireduct
$(X, B)$ with $|B| \geq 2$ can be transformed into another correct ensemble of no greater
description length by replacing that bireduct with a collection of simpler bireducts (each having
at most one attribute).

## Statement

Let $\mathcal{B} = \{(X_1, B_1), \ldots, (X_m, B_m)\}$ be a correct ensemble for
$\mathbb{A}_{\mathcal{S}}$, containing a decision bireduct $(X_i, B_i)$ with $|B_i| \geq 2$.
Define the replacement multiset:

$$
\mathcal{R} = \{(X_b, \{b\}) : b \in B_i\} \cup
\{(U_{\mathcal{S}}, \emptyset) : k = 1, \ldots, |B_i| - 1\},
$$

where each $X_b \subseteq U_{\mathcal{S}} \cup \{u_*\}$ is determined in accordance with
[prop-solution-bireduct-properties](solution-bireduct-properties.md). Then, the ensemble:

$$
\mathcal{B}' = \{(X_k, B_k) \in \mathcal{B} : k \neq i\} \cup \mathcal{R}
$$

is correct and $EnsDesc(\mathcal{B}') \leq EnsDesc(\mathcal{B})$.

## Proof

**Size of the new ensemble.** After the replacement:

$$
\begin{aligned}
|\mathcal{B}'| &= |\mathcal{B}| - 1 + |\{(X_b, \{b\}) : b \in B_i\}| +
|\{(U_{\mathcal{S}}, \emptyset) : k = 1, \ldots, |B_i| - 1\}| \\
&= |\mathcal{B}| - 1 + |B_i| + |B_i| - 1 \\
&= |\mathcal{B}| + 2 \cdot |B_i| - 2.
\end{aligned}
$$

**Correctness for the special object $u_*$.** From the correctness of $\mathcal{B}$:
$cov_{\mathcal{B}}(u_*) > |\mathcal{B}|/2$.
From [prop-solution-bireduct-properties](solution-bireduct-properties.md), $u_* \in X_i$. In
$\mathcal{B}'$, the bireduct $(X_i, B_i)$ is removed (loss of 1 in the coverage count), but all
$|B_i|$ single-attribute bireducts $(X_b, \{b\})$ cover $u_*$ (gain of $|B_i|$). The empty-attribute
bireducts $(U_{\mathcal{S}}, \emptyset)$ do not cover $u_*$ (since $u_* \notin U_{\mathcal{S}}$).
Therefore:

$$
\begin{aligned}
cov_{\mathcal{B}'}(u_*) &= cov_{\mathcal{B}}(u_*) - 1 + |B_i| \\
&> \frac{|\mathcal{B}|}{2} + |B_i| - 1 \\
&= \frac{|\mathcal{B}| + 2 \cdot |B_i| - 2}{2}
= \frac{|\mathcal{B}'|}{2}.
\end{aligned}
$$

**Correctness for non-special objects $u_\omega \in U_{\mathcal{S}}$.** From
[prop-solution-bireduct-properties](solution-bireduct-properties.md), there are two possibilities.

*Case 1: $u_\omega$ has at least one $1$ for attributes in $B_i$.* Then $u_\omega \in X_i$ and it
will no longer be covered by the replaced bireduct (loss of 1), but it is covered by all $|B_i|-1$
empty-attribute bireducts $(U_{\mathcal{S}}, \emptyset)$ and at least one single-attribute bireduct
$(X_b, \{b\})$:

$$
\begin{aligned}
cov_{\mathcal{B}'}(u_\omega) &\geq cov_{\mathcal{B}}(u_\omega) - 1 + (|B_i| - 1) + 1 \\
&> \frac{|\mathcal{B}|}{2} + |B_i| - 1
= \frac{|\mathcal{B}'|}{2}.
\end{aligned}
$$

*Case 2: $u_\omega$ has all $0$s for attributes in $B_i$.* Then $u_\omega \notin X_i$, so it was
not covered by the replaced bireduct (no loss). It is covered by all $|B_i|-1$ empty-attribute
bireducts but none of the single-attribute bireducts:

$$
\begin{aligned}
cov_{\mathcal{B}'}(u_\omega) &= cov_{\mathcal{B}}(u_\omega) + (|B_i| - 1) \\
&> \frac{|\mathcal{B}|}{2} + |B_i| - 1
= \frac{|\mathcal{B}'|}{2}.
\end{aligned}
$$

In both cases $cov_{\mathcal{B}'}(u_\omega) > |\mathcal{B}'|/2$. Together with the argument for
$u_*$, $\mathcal{B}'$ is a correct ensemble.

**Description length comparison.** From
[prop-bireduct-desc-len-geq-bplus1-squared](../../propositions/bireduct-desc-len-geq-bplus1-squared.md):
$BirDesc(X_i, B_i) \geq (|B_i| + 1)^2$.
From [prop-bireducts-0-and-1-attrs-desc-size](bireducts-0-and-1-attrs-desc-size.md), the
replacement bireducts have fixed lengths: $BirDesc(X_b, \{b\}) = 4$ and
$BirDesc(U_{\mathcal{S}}, \emptyset) = 1$. Therefore:

$$
\begin{aligned}
EnsDesc(\mathcal{R}) &= \sum_{b \in B_i} BirDesc(X_b, \{b\}) +
\sum_{k=1}^{|B_i|-1} BirDesc(U_{\mathcal{S}}, \emptyset) \\
&= |B_i| \cdot 4 + (|B_i| - 1) \cdot 1 = 5 \cdot |B_i| - 1.
\end{aligned}
$$

The change in description length is bounded by:

$$
\begin{aligned}
\Delta &= EnsDesc(\mathcal{R}) - BirDesc(X_i, B_i) \\
&\leq 5 \cdot |B_i| - 1 - (|B_i| + 1)^2 \\
&= -|B_i|^2 + 3 \cdot |B_i| - 2.
\end{aligned}
$$

This quadratic function has a negative leading coefficient and roots at $|B_i| = 1$ and
$|B_i| = 2$. For the considered $|B_i| \geq 2$, its value is non-positive. Hence $EnsDesc(\mathcal{B}') \leq EnsDesc(\mathcal{B})$.

## Consequences

The replacement procedure can be applied repeatedly. Starting from any correct ensemble
$\mathcal{B}$, each bireduct $(X_i, B_i)$ with $|B_i| \geq 2$ can be replaced by the collection
$\mathcal{R}$. Each replacement preserves correctness and does not increase the description length.
After finitely many steps, all bireducts have at most one attribute. Therefore, if a correct
ensemble with a given description length exists, there is also a correct ensemble consisting
exclusively of bireducts with 0 or 1 attributes whose overall description length is no greater.
