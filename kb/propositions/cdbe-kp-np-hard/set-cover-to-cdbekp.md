---
id: prop-set-cover-to-cdbekp
type: proposition
status: complete
created: 2026-09-21
updated: 2026-09-21
tags: [complexity, bireducts, ensemble]
requires:
  - prop-set-cover-construction
  - prop-0-1-bireduct-existence
  - prop-bireducts-0-and-1-attrs-desc-size
  - concept-bireduct-ensemble
  - concept-decision-bireduct
see_also:
  - prop-cdbe-kp-np-hard
  - prop-bireduct-replacement
source: src-reduct-matrix-ensembles
---

# From a Set Cover to a Correct Bireduct Ensemble

A set cover of size $m$ over the instance $\mathcal{I} = (W, \mathcal{S})$ yields a correct
ensemble of decision bireducts for $\mathbb{A}_{\mathcal{I}}$ with description length exactly
$5m - 1$. This is the forward direction of the NP-hardness reduction from Set Cover to CDBEkP.

## Statement

Let $\mathcal{I} = (W, \mathcal{S})$ be a Set Cover instance with
$\mathcal{S} = \{S_1, \dots, S_n\}$ and $\bigcup \mathcal{S} = W$, where $W \neq \emptyset$. If
$\mathcal{C} \subseteq \mathcal{S}$ is a cover of size $m$ consisting entirely of non-empty sets,
then there exists a correct ensemble for the decision table $\mathbb{A}_{\mathcal{I}}$ constructed
in [prop-set-cover-construction](set-cover-construction.md) that has description length equal to
$5m - 1$.

## Proof

Let $\mathcal{C} \subseteq \mathcal{S}$ be a cover with $|\mathcal{C}| = m$. Construct the
0-1-bireduct ensemble $\mathcal{B} = \mathcal{L} \cup \mathcal{M}$, where

$$
\mathcal{L} = \{(U_{\mathcal{I}}, \emptyset) \times (m - 1)\}
$$

and

$$
\mathcal{M} = \{(X_{a_{S_i}}, \{a_{S_i}\}) : S_i \in \mathcal{C}\},
\qquad
X_{a_{S_i}} = \{u_*\} \cup \{u_\omega \in U_{\mathcal{I}} : a_{S_i}(u_\omega) = 1\}.
$$

By [prop-0-1-bireduct-existence](0-1-bireduct-existence.md), each $(X_{a_{S_i}}, \{a_{S_i}\})$
(since $S_i \in \mathcal{C}$ is non-empty) and each copy of $(U_{\mathcal{I}}, \emptyset)$ is a
decision bireduct, so $\mathcal{B}$ is an ensemble of decision bireducts. We have
$|\mathcal{B}| = (m - 1) + m = 2m - 1$.

**Coverage of the special object $u_*$.** The copies of $(U_{\mathcal{I}}, \emptyset)$ do not
cover $u_*$ (as $u_* \notin U_{\mathcal{I}}$), while every $(X_{a_{S_i}}, \{a_{S_i}\})$ contains
$u_*$. Hence

$$
cov_{\mathcal{B}}(u_*) = 0 + m = m > m - \tfrac{1}{2} = \tfrac{|\mathcal{B}|}{2}.
$$

**Coverage of $u_\omega \in U_{\mathcal{I}}$.** Every copy of $(U_{\mathcal{I}}, \emptyset)$
covers $u_\omega$, contributing $m - 1$. Since $\mathcal{C}$ covers $W$, there is some $S_i \in
\mathcal{C}$ with $\omega \in S_i$, hence $a_{S_i}(u_\omega) = 1$ and $u_\omega \in X_{a_{S_i}}$,
so $\mathcal{M}$ contributes at least $1$. Therefore

$$
cov_{\mathcal{B}}(u_\omega) \geq (m - 1) + 1 = m > m - \tfrac{1}{2} = \tfrac{|\mathcal{B}|}{2}.
$$

Thus $\mathcal{B}$ is a correct ensemble.

**Description length.** By
[prop-bireducts-0-and-1-attrs-desc-size](bireducts-0-and-1-attrs-desc-size.md),
$BirDesc(U_{\mathcal{I}}, \emptyset) = 1$ and $BirDesc(X_{a_{S_i}}, \{a_{S_i}\}) = 4$. Hence

$$
EnsDesc(\mathcal{B}) = 1 \cdot (m - 1) + 4 \cdot m = 5m - 1.
$$
