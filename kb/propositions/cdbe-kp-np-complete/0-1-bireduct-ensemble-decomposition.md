---
id: prop-0-1-bireduct-ensemble-decomposition
type: proposition
status: complete
created: 2026-07-14
updated: 2026-07-17
tags: [complexity, bireducts, ensemble]
requires:
  - prop-set-cover-construction
  - prop-solution-bireduct-properties
  - concept-bireduct-ensemble
see_also:
  - prop-bireduct-replacement
  - prop-cdbe-kp-np-complete
  - prop-correct-ensemble-m-nonempty
  - prop-correct-ensemble-remove-k
source: "Slezak & Stawicki, 'Complexity of Searching for the Simplest Reduct Matrix Ensembles'
  (paper in preparation)"
---

# 0-1-Bireduct Ensemble Decomposition

Any 0-1-bireduct ensemble for the transformed table $\mathbb{A}_{\mathcal{S}}$ can be expressed
uniquely as a union of three multisets of bireducts corresponding to the three structurally
distinct categories of bireducts that can appear in such an ensemble.

## Statement

Let $\mathcal{B}$ be a 0-1-bireduct ensemble for $\mathbb{A}_{\mathcal{S}}$. Then $\mathcal{B}$
can be represented uniquely as a union of three multisets of bireducts:

$$
\mathcal{B} = \mathcal{K} \cup \mathcal{L} \cup \mathcal{M},
$$

where $\mathcal{K}, \mathcal{L}, \mathcal{M}$ are defined as follows:

$$
\begin{aligned}
\mathcal{K} &= \{(\{u_*\}, \emptyset) \times K\} \\
\mathcal{L} &= \{(U_{\mathcal{S}}, \emptyset) \times L\} \\
\mathcal{M} &= \{(X_b, \{b\}) \times M_b : b \in B\}.
\end{aligned}
$$

Here $K$ and $L$ are non-negative integers, while $B \subseteq A_{\mathcal{S}}$ is the set of
attributes appearing in the single-attribute bireducts and each $M_b$ is a positive integer. The
notation "$(\cdot) \times n$" denotes $n$ copies of that element in a multiset.

## Proof

We know that $\mathcal{B}$ consists of bireducts with attribute subsets of cardinality only zero
or one. Hence, we can distinguish three disjoint categories of bireducts and justify that each
bireduct in $\mathcal{B}$ belongs to exactly one of them. For an empty subset of attributes,
there are only two options (see
[the structural lemma](solution-bireduct-properties.md)) for what such a bireduct in
$\mathbb{A}_{\mathcal{S}}$ might look like, i.e., it can be either
$(\{u_*\}, \emptyset)$ or $(U_{\mathcal{S}}, \emptyset)$. We will consider these as two disjoint
categories $\mathcal{K}$ and $\mathcal{L}$, respectively. Furthermore, $K$ and $L$ denote the
number of copies of these bireducts in $\mathcal{B}$. Let the third category $\mathcal{M}$ gather
all other bireducts that have a single-element subset of attributes. $\mathcal{M}$ can be
expressed as a multiset where each attribute $b$ used in some bireduct from $\mathcal{B}$ has its
corresponding multiplicity $M_b$. Hence, the initial ensemble can be represented as follows:

$$
\mathcal{B} =
\underbrace{\{(\{u_*\}, \emptyset) \times K\}}_{\substack{\mathcal{K}\\|\mathcal{K}| = K}}
\;\cup\;
\underbrace{\{(U_{\mathcal{S}}, \emptyset) \times L\}}_{\substack{\mathcal{L}\\|\mathcal{L}| = L}}
\;\cup\;
\underbrace{\{(X_b, \{b\}) \times M_b : b \in B\}}_{\substack{\mathcal{M}\\|\mathcal{M}| = \sum_{b \in B} M_b = M}}
$$

Since the three categories are disjoint and each decision bireduct from $\mathcal{B}$ falls into
exactly one, the decomposition is unique. ∎

## Consequences

The K/L/M decomposition is the foundation for the subsequent lemmas in the CDBEkP
NP-completeness reduction. In particular:

- The decomposition enables analysis of coverage counts per category: $\mathcal{K}$ covers only
  $u_*$, $\mathcal{L}$ covers only $U_{\mathcal{S}}$, $\mathcal{M}$ covers both.
- A correct 0-1-bireduct ensemble must have $\mathcal{M} \neq \emptyset$ (see
  [prop-correct-ensemble-m-nonempty](correct-ensemble-m-nonempty.md)).
