---
id: prop-relative-m-epsilon-reduct-np-hard
type: proposition
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [complexity, np-hardness, reduction, approximate-reducts, majority-function]
requires:
  [concept-approximate-decision-reduct,
   concept-majority-function,
   concept-np-hardness-foundations,
   prop-alpha-dominating-set-np-hard]
see_also:
  [prop-alpha-dominating-set-np-hard,
   prop-m-epsilon-reduct-np-hard,
   prop-relative-r-epsilon-reduct-np-hard,
   concept-approximate-decision-reduct,
   concept-majority-function]
source: src-thesis-phd
---

# NP-Hardness of Minimal Relative M-Decision Epsilon-Reduct

For any $\varepsilon \in [0, 1)$, finding a minimal relative $M$-decision $\varepsilon$-reduct is
NP-hard via a polynomial reduction from the Minimal $\alpha$-Dominating Set problem.

## Statement

Let $\varepsilon \in [0, 1)$ be given. The problem of finding a minimal relative $M$-decision
$\varepsilon$-reduct for a decision table $\mathbb{A} = (U, A \cup \{d\})$ is NP-hard.

## Proof Strategy

Construct a decision table from a graph such that the majority function $M(B)$ for any attribute
subset $B$ can be expressed in terms of the dominating set $\mathbb{W}_B = \{v_j : a_j \in B\}$
induced by $B$. Then derive a formula linking $\varepsilon$ and $\alpha$ such that the relative
$M$-superreduct condition is equivalent to the $\alpha$-dominating set condition. Minimality is
then preserved across the reduction.

## Proof

The stated problem has been addressed in Slezak, D. (2000). *Normalized Decision Functions and
Measures for Inconsistent Decision Tables Analysis*. Fundamenta Informaticae, 44(3), 291-319.
The following summarizes the ideas for completeness of the dissertation.

A polynomial reduction from the problem of finding a minimal $\alpha$-dominating set (NP-hard per
[NP-Hardness of the Minimal Alpha-Dominating Set Problem](alpha-dominating-set-np-hard.md)) to the
problem of finding a minimal relative $M$-decision $\varepsilon$-reduct is shown.

### Construction of the Decision Table

For a given graph $\mathbb{G} = (\mathbb{V}, \mathbb{E})$, parametrize the transformation by
$\varepsilon$ and construct the decision table
$\mathbb{A}^{\varepsilon*}_{\mathbb{G}} = (U^{\varepsilon*}_{\mathbb{G}},
A^{\varepsilon*}_{\mathbb{G}} \cup \{d^{\varepsilon*}_{\mathbb{G}}\})$.

Let $m(\varepsilon) = \lfloor (1 - \varepsilon)^{-1} + 1 \rfloor$.

**Objects.** For each vertex $v_i \in \mathbb{V}$, create $m(\varepsilon)$ objects:
$U^{\varepsilon*}_{\mathbb{G}} = \{u_1, \ldots, u_{m(\varepsilon)|\mathbb{V}|}\}$.
Define the chunk helper: $\operatorname{chunk}(i) = \lceil i / m(\varepsilon) \rceil$.

**Conditional attributes.** One per vertex:
$A^{\varepsilon*}_{\mathbb{G}} = \{a_1, a_2, \ldots, a_{|\mathbb{V}|}\}$.

**Decision attribute.** Values in $\{1, \ldots, m(\varepsilon)\}$:
$$
d^{\varepsilon*}_{\mathbb{G}}(u_i) = i - (\operatorname{chunk}(i) - 1)m(\varepsilon).
$$

This assigns the $m(\varepsilon)$ objects corresponding to each vertex distinct decision values
$1, 2, \ldots, m(\varepsilon)$.

**Conditional attribute values.** For each $a_j$:
$$
a_j(u_i) =
\begin{cases}
i & \text{if } \operatorname{chunk}(i) = j \lor (v_{\operatorname{chunk}(i)}, v_j) \in \mathbb{E},\\
(\operatorname{chunk}(i) - 1)m(\varepsilon) + 1 & \text{otherwise}.
\end{cases}
$$

The table $\mathbb{A}^{\varepsilon*}_{\mathbb{G}}$ is consistent.

### Majority Function Formula

For any $B \subseteq A^{\varepsilon*}_{\mathbb{G}}$, define $\mathbb{W}_B = \{v_j \in \mathbb{V} :
a_j \in B\}$. It was shown in the cited work that:

$$
M(B) = \frac{|Cov_{\mathbb{G}}(\mathbb{W}_B)| \cdot m(\varepsilon) + |\mathbb{V} \setminus
Cov_{\mathbb{G}}(\mathbb{W}_B)|}{|U^{\varepsilon*}_{\mathbb{G}}|}.
$$

Intuition: the majority function counts correct classifications. An object $u_i$ with
$\operatorname{chunk}(i)$ corresponding to a vertex $v$ in $Cov_{\mathbb{G}}(\mathbb{W}_B)$ is
classified correctly by $B$ (because some $a_j \in B$ discerns it from other decision classes),
contributing $m(\varepsilon)$ correct objects. Non-covered vertices contribute only 1 correctly
classified object (the default decision when no attribute distinguishes classes).

### Linking epsilon and alpha

Define:

$$
\alpha(\varepsilon) = 1 - \frac{\varepsilon}{1 - m(\varepsilon)^{-1}}.
$$

For $\varepsilon \in [0, 1)$, one can verify that $\alpha(\varepsilon) \in (0, 1]$, so the
Minimal $\alpha(\varepsilon)$-Dominating Set problem is NP-hard.

The key correspondence proved in the cited work: in $\mathbb{A}^{\varepsilon*}_{\mathbb{G}}$,

$$
M(B) \geq (1 - \varepsilon) M(A^{\varepsilon*}_{\mathbb{G}})
$$
holds if and only if
$$
\frac{|Cov_{\mathbb{G}}(\mathbb{W}_B)|}{|\mathbb{V}|} \geq \alpha(\varepsilon)
$$
holds in $\mathbb{G}$.

Thus $B$ is a relative $M$-decision $\varepsilon$-superreduct for $\mathbb{A}^{\varepsilon*}_{\mathbb{G}}$
if and only if $\mathbb{W}_B$ is an $\alpha(\varepsilon)$-dominating set for $\mathbb{G}$.

### Minimality Correspondence

Analogous to the $\gamma$ case, a minimal relative $M$-decision $\varepsilon$-reduct $B$ must
correspond to a minimal $\alpha(\varepsilon)$-dominating set $\mathbb{W}_B$: if a smaller
dominating set $\mathbb{W}'$ existed, its corresponding $B' = \{a_j : v_j \in \mathbb{W}'\}$ would
be a smaller $M$-superreduct, contradicting the minimality of $B$.

Therefore a polynomial-time algorithm for finding a minimal relative $M$-decision
$\varepsilon$-reduct would yield a polynomial-time algorithm for the Minimal
$\alpha(\varepsilon)$-Dominating Set problem, contradicting its NP-hardness.
