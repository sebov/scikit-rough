---
id: prop-solution-bireduct-properties
type: proposition
status: draft
created: 2026-06-21
updated: 2026-06-21
tags: [complexity, bireducts, ensemble]
requires:
  - prop-set-cover-construction
  - prop-transformed-table-consistent
  - concept-decision-bireduct
  - concept-indiscernibility
see_also:
  - prop-cdbe-kp-np-complete
  - concept-bireduct-ensemble
source: src-reduct-matrix-ensembles
---

# Structural Properties of Bireducts in the Transformed Table

Decision bireducts in $\mathbb{A}_{\mathcal{S}}$ have a rigid two-case structure determined
solely by whether their attribute set $B$ is empty or non-empty. This lemma is referenced
throughout the remainder of the NP-completeness proof.

## Statement

Let $(W, \mathcal{S})$ be a Set Cover instance with $\bigcup \mathcal{S} = W$, let
$\mathbb{A}_{\mathcal{S}}$ be the decision table constructed in
[the construction](set-cover-construction.md), and let $(X, B)$ be a decision bireduct for
$\mathbb{A}_{\mathcal{S}}$.

1. If $|B| \geq 1$, then
   $$X = \{u_*\} \cup \{\, u_\omega \in U_{\mathcal{S}} \mid
   a_{S_i}(u_\omega) = 1 \text{ for some } a_{S_i} \in B \,\}.$$

2. If $B = \emptyset$, then $X = \{u_*\}$ or $X = U_{\mathcal{S}}$.

## Proof

**(1)** For a decision bireduct $(X, B)$ with $|B| \geq 1$, if every $a_{S_i} \in B$ corresponds
to an empty subset $S_i = \emptyset$, then all objects take value $0$ on every attribute in $B$
and $IND(B) = IND(\emptyset)$. That would contradict the irreducibility of $B$ in the decision
bireduct $(X, B)$. Hence, at least one $a_{S_i} \in B$ corresponds to a non-empty $S_i \in
\mathcal{S}$, and there are at least two indiscernibility classes in
$\mathbb{A}_{\mathcal{S}}$: the one containing $u_*$ (and possibly some other objects), and
the remaining classes, where objects have value $1$ for at least one attribute in $B$. The
latter classes contain only objects with decision value $0$. Objects from these classes must
be entirely contained in $X$: if any were missing, it could be added without breaking the
functional dependency $B \Rrightarrow_X d$, since they share decision $0$ and are already
discernible from $u_*$ (the only object with decision $1$). This would contradict the
maximality of $X$ in the decision bireduct.

In the class containing $u_*$, diversity in decision values is possible -- $u_*$ has decision
$1$, but there may be other objects with decision $0$ (depending on the original input and
the specific $B$). Although it may appear that there is freedom of choice from that class,
either $u_*$ or the other objects with decision value $0$, in fact only $u_*$ must be chosen.
If instead the objects with decision $0$ from this class were chosen, the whole $X$ would
contain only objects with decision $0$, and $B$ could be reduced to $\emptyset$, contradicting
the assumption that $(X, B)$ is a decision bireduct. Thus $X$ consists exactly of $u_*$ and
all objects from $U_{\mathcal{S}}$ that have at least one $1$ on attributes from $B$.

**(2)** If $B = \emptyset$, there is a single indiscernibility class containing all objects.
Since $X$ must be a maximal subset with uniform decision value, we have only two options:
$X = \{u_*\}$ (decision value $1$) or $X = U_{\mathcal{S}}$ (decision value $0$).

## Remarks

This proposition is the key structural lemma for the reduction. It implies that every bireduct
with $|B| \geq 1$ is uniquely determined by $B$. The two cases are used throughout the
subsequent lemmas on ensemble decomposition into $K$, $L$, and $M$ multisets.

In the language of the original Set Cover instance, the condition
$a_{S_i}(u_\omega) = 1$ for some $a_{S_i} \in B$ is equivalent to
$\omega \in \bigcup_{a_{S_i} \in B} S_i$. Hence for $|B| \geq 1$, the object set of the
bireduct corresponds exactly to the special object together with all elements of $W$ that are
covered by the subfamily of $\mathcal{S}$ indexed by $B$.

In particular, an attribute $a_{S_i}$ with $S_i = \emptyset$ cannot belong to any decision
bireduct, as the corresponding column would consist entirely of zeros and such an attribute
would be reducible to $\emptyset$.
