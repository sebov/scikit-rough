---
id: prop-correct-ensemble-m-nonempty
type: proposition
status: complete
created: 2026-07-17
updated: 2026-07-17
tags: [complexity, bireducts, ensemble]
requires:
  - prop-0-1-bireduct-ensemble-decomposition
  - prop-set-cover-construction
  - concept-bireduct-ensemble
see_also:
  - prop-0-1-bireduct-ensemble-decomposition
  - prop-cdbe-kp-np-complete
  - prop-bireduct-replacement
source: "Slezak & Stawicki, 'Complexity of Searching for the Simplest Reduct Matrix Ensembles'
  (paper in preparation)"
---

# Correct 0-1-Bireduct Ensemble Must Contain a Single-Attribute Bireduct

A correct 0-1-bireduct ensemble for the transformed table $\mathbb{A}_{\mathcal{S}}$ must contain
at least one bireduct with exactly one attribute (i.e., $\mathcal{M} \neq \emptyset$).

## Statement

Let $(W, \mathcal{S})$ be a Set Cover instance with $\bigcup \mathcal{S} = W$ and
$W \neq \emptyset$, let $\mathbb{A}_{\mathcal{S}}$ be the decision table constructed in
[prop-set-cover-construction](set-cover-construction.md), and let $\mathcal{B}$ be a correct
0-1-bireduct ensemble for $\mathbb{A}_{\mathcal{S}}$. Then $\mathcal{B}$ must contain at least one
bireduct with exactly one attribute.

## Proof

By [prop-0-1-bireduct-ensemble-decomposition](0-1-bireduct-ensemble-decomposition.md), we can
uniquely write $\mathcal{B}$ as $\mathcal{B} = \mathcal{K} \cup \mathcal{L} \cup \mathcal{M}$,
where $\mathcal{K} = \{(\{u_*\}, \emptyset) \times K\}$,
$\mathcal{L} = \{(U_{\mathcal{S}}, \emptyset) \times L\}$, and
$\mathcal{M} = \{(X_b, \{b\}) \times M_b : b \in B\}$ for some subset of attributes
$B \subseteq A_{\mathcal{S}}$. Given that $\mathcal{B}$ is correct, it follows that
$\forall u \in U_{\mathcal{S}} \cup \{u_*\} \;
cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}}(u) > |\mathcal{B}|/2$ (see
[concept-bireduct-ensemble](../../concepts/bireduct-ensemble.md)).

Let us assume that $\mathcal{M} = \emptyset$. Thus, $\mathcal{B} = \mathcal{K} \cup \mathcal{L}$.
Then the only bireducts in $\mathcal{B}$ that cover $u_*$ are the copies of
$(\{u_*\}, \emptyset)$, while the only ones covering any object from $U_{\mathcal{S}}$ are copies
of $(U_{\mathcal{S}}, \emptyset)$. In consequence, combined with the correctness condition:

$$
cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}}(u_*) = |\mathcal{K}|
  > \frac{|\mathcal{B}|}{2} = \frac{|\mathcal{K}| + |\mathcal{L}|}{2}
$$

and, for any $u \in U_{\mathcal{S}}$ (which is non-empty since $W \neq \emptyset$):

$$
cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}}(u) = |\mathcal{L}|
  > \frac{|\mathcal{B}|}{2} = \frac{|\mathcal{K}| + |\mathcal{L}|}{2}
$$

Two numbers cannot both strictly exceed their average -- a contradiction. Hence
$\mathcal{M} \neq \emptyset$, so $\mathcal{B}$ must contain at least one bireduct with exactly one
attribute. ∎

## Consequences

This lemma is used in the second step of the CDBEkP NP-completeness reduction: after decomposing a
correct 0-1-bireduct ensemble into $\mathcal{K} \cup \mathcal{L} \cup \mathcal{M}$, the non-emptiness
of $\mathcal{M}$ guarantees the existence of at least one single-attribute bireduct, which
corresponds to a selected set in the Set Cover solution. The next lemma shows that $\mathcal{K}$
can be entirely removed from the ensemble while preserving correctness.
