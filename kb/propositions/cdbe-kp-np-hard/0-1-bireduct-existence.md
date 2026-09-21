---
id: prop-0-1-bireduct-existence
type: proposition
status: complete
created: 2026-09-21
updated: 2026-09-21
tags: [complexity, bireducts, ensemble]
requires:
  - prop-set-cover-construction
  - concept-decision-bireduct
see_also:
  - prop-solution-bireduct-properties
  - prop-cdbe-kp-np-hard
source: src-reduct-matrix-ensembles
---

# Existence of 0-1 Decision Bireducts in the Transformed Table

The transformed table $\mathbb{A}_{\mathcal{I}}$ admits decision bireducts with an empty or a
single-attribute set. This proposition is the sufficiency companion to
[prop-solution-bireduct-properties](solution-bireduct-properties.md): that lemma states that every
bireduct must have one of the shapes below, whereas this proposition states that these shapes are
indeed decision bireducts.

## Statement

Let $\mathcal{I} = (W, \mathcal{S})$ be a Set Cover instance with
$\mathcal{S} = \{S_1, \dots, S_n\}$ and $\bigcup \mathcal{S} = W$, where $W \neq \emptyset$, and let
$\mathbb{A}_{\mathcal{I}}$ be the decision table constructed in
[prop-set-cover-construction](set-cover-construction.md). The following pairs are decision
bireducts for $\mathbb{A}_{\mathcal{I}}$:

1. $(\{u_*\}, \emptyset)$,
2. $(U_{\mathcal{I}}, \emptyset)$,
3. for every non-empty $S_i \in \mathcal{S}$,
   $$(\{u_*\} \cup \{u_\omega \in U_{\mathcal{I}} : a_{S_i}(u_\omega) = 1\},\ \{a_{S_i}\}).$$

## Proof

We verify the three conditions of a decision bireduct.

**(1)** For $(\{u_*\}, \emptyset)$: determination holds vacuously, as a single object yields no
pair of objects with distinct decisions; irreducibility holds vacuously, as $\emptyset$ has no
proper subset; maximality holds because every other object has decision value
$0 \neq 1 = d_{\mathcal{I}}(u_*)$, which $\emptyset$ cannot discern.

**(2)** For $(U_{\mathcal{I}}, \emptyset)$: all objects in $U_{\mathcal{I}}$ have decision value
$0$, so determination is trivial and $\emptyset$ is irreducible. The only proper superset is
$U_{\mathcal{I}} \cup \{u_*\}$, which introduces a mixed-decision pair that $\emptyset$ cannot
discern; hence $U_{\mathcal{I}}$ is maximal.

**(3)** For a non-empty $S_i$, put $X = \{u_*\} \cup \{u_\omega \in U_{\mathcal{I}} :
a_{S_i}(u_\omega) = 1\}$ and $B = \{a_{S_i}\}$. The object $u_*$ is the only one in $X$ with
decision value $1$, and it differs from every other object in $X$ on $a_{S_i}$ ($0$ vs. $1$), so
all mixed-decision pairs are discerned and determination holds. The only proper subset of $B$ is
$\emptyset$, which fails to determine $d_{\mathcal{I}}$ because $X$ contains both decision values
(guaranteed by $S_i$ being non-empty); hence $B$ is irreducible. Finally, every object
$u_\omega \notin X$ satisfies $a_{S_i}(u_\omega) = 0$, so it is indiscernible from $u_*$ by $B$
while having the opposite decision value; therefore $X$ cannot be extended without breaking
determination.

## Remarks

Together with [prop-solution-bireduct-properties](solution-bireduct-properties.md), this fully
characterizes the 0-1 decision bireducts of $\mathbb{A}_{\mathcal{I}}$. The sufficiency does not
extend to $|B| \geq 2$: for a larger $B$, the pair determined by the necessity lemma is not always
a bireduct, because $B$ may contain redundant attributes.
