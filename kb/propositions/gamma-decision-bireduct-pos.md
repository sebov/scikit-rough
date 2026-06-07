---
id: prop-gamma-decision-bireduct-pos
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [core, bireducts, positive-region]
requires:
  [concept-gamma-decision-bireduct,
   concept-positive-region,
   concept-indiscernibility]
see_also:
  [concept-gamma-decision-bireduct,
   concept-gamma-decision-reduct,
   prop-gamma-decision-bireduct-to-reduct]
source: src-thesis-phd
---

# Gamma-Decision Bireduct Equals Positive Region

A $\gamma$-decision bireduct $(X, B)$ is completely characterized by the positive region: $X$ must
equal $POS(B)$, and $B$ must be irreducible with respect to preserving the positive region.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. Let subsets $B \subseteq A$ and $X \subseteq U$ be
given. Then $(X, B)$ is a $\gamma$-decision bireduct if and only if the following two properties
hold:

1. $X = POS(B)$.
2. There is no proper subset $B' \subsetneq B$ such that $POS(B') = POS(B)$.

## Proof

**($\Leftarrow$)** We need to check three requirements for a $\gamma$-decision bireduct.

Let us first show that $B \Rrightarrow^{\gamma}_{POS(B)} d$ holds. Suppose that
$B \not\Rrightarrow^{\gamma}_{POS(B)} d$. Following the definition of $\gamma$-decision bireduct,
there would exist a pair of objects $u_i \in POS(B)$, $u_j \in U$ which is not discerned by $B$
and such that $d(u_i) \neq d(u_j)$. However, this would mean that $u_j \in [u_i]_B$, so --
according to the definition of positive region -- $u_i$ could not belong to $POS(B)$, a
contradiction.

Let us now proceed with the third requirement in the definition of $\gamma$-decision bireduct
(object maximality). If $B \Rrightarrow^{\gamma}_{X'} d$ held for some $X' \supsetneq POS(B)$, then
$POS(B)$ could be extended -- a contradiction. Specifically, any object in $X' \setminus POS(B)$
would have its entire $B$-equivalence class contained in a single decision class (by the
gamma-determination), which means it should belong to $POS(B)$.

Finally, consider the second requirement for $(X, B)$ to be a $\gamma$-decision bireduct (attribute
irreducibility). By following the same reasoning as above, we know that for any $B' \subsetneq B$
we can have $B' \not\Rrightarrow^{\gamma}_{X'} d$ only for subsets $X' \subseteq POS(B')$. Given
$POS(B') \subsetneq POS(B)$ (from condition 2), we also know that there is no $B' \subsetneq B$
such that $B' \Rrightarrow^{\gamma}_{POS(B)} d$ holds.

**($\Rightarrow$)** Suppose that $X \neq POS(B)$. This means that either $POS(B) \setminus X \neq
\emptyset$ or $X \setminus POS(B) \neq \emptyset$.

In the first case, we would have $u \in POS(B) \setminus X$. From the definition of positive
region, for $u \in POS(B)$ all objects in $[u]_B$ have the same value for $d$. However, this means
that the whole $[u]_B$ should be included in $X$ because, otherwise, $X$ would not be
non-extendable (we could add $[u]_B$ to $X$ without violating gamma-determination).

In the second case, we would have $u \in X \setminus POS(B)$. It would mean that $[u]_B$ contains
objects from more than one decision class (because otherwise $u$ would be in $POS(B)$). This also
leads to contradiction because we would then have at least one pair of objects $u_i = u \in X$ and
$u_j \in [u]_B$ not discerned by $B$, such that $d(u_i) \neq d(u_j)$, thus violating the
definition of a $\gamma$-decision bireduct.

Hence, $X = POS(B)$.

Now, suppose that there exists $B' \subsetneq B$ such that $POS(B) = POS(B')$. As we have just
shown above, we know that $B' \Rrightarrow^{\gamma}_{POS(B')} d$. Therefore, because
$X = POS(B) = POS(B')$, we would have that $B' \Rrightarrow^{\gamma}_X d$. It would contradict the
requirement of irreducibility of $B$ in the definition of $\gamma$-decision bireduct.

## Consequences

This proposition leads to several important observations:

**Uniqueness.** For a given $B$, there can be at most one $\gamma$-decision bireduct with that
attribute subset. If $(X, B)$ and $(X', B)$ were both $\gamma$-decision bireducts, then
$X = POS(B) = X'$, so they must be identical.

**Reduction to standard reducts.** The problem of searching for $\gamma$-decision bireducts in
$\mathbb{A}$ can be transformed to the problem of searching decision reducts in the modified
consistent table $\mathbb{A}_A^\gamma$, because the gamma-modification leaves objects from
$POS(A)$ unchanged.

**Rule interpretation.** For a $\gamma$-decision bireduct $(X, B)$ where $X = POS(B)$, the rules
with supports summing to $X$ are all deterministic with respect to the whole $U$:

$$
Rules_{\gamma}(POS(B), B) =
\left\{
  \bigwedge_{a \in B} (a = a(u)) \Rightarrow (d = d(u)) : u \in POS(B)
\right\}
$$
