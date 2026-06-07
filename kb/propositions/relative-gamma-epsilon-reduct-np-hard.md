---
id: prop-relative-gamma-epsilon-reduct-np-hard
type: proposition
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [complexity, np-hardness, reduction, approximate-reducts]
requires:
  [concept-approximate-decision-reduct,
   concept-positive-region,
   concept-np-hardness-foundations]
see_also:
  [prop-minimal-dominating-set-np-hard,
   prop-gamma-epsilon-reduct-np-hard,
   prop-alpha-dominating-set-np-hard,
   concept-approximate-decision-reduct,
   concept-np-hardness-foundations]
source: tmp/phd/thesis.tex
---

# NP-Hardness of Minimal Relative Gamma-Decision Epsilon-Reduct

For any $\varepsilon \in [0, 1)$, finding a minimal relative $\gamma$-decision $\varepsilon$-reduct is
NP-hard via a polynomial reduction from the Minimal Dominating Set problem.

## Statement

Let $\varepsilon \in [0, 1)$ be given. The problem of finding a minimal relative $\gamma$-decision
$\varepsilon$-reduct for an input decision table is NP-hard.

## Proof Strategy

Construct a decision table $\mathbb{A}^{\varepsilon}_{\mathbb{G}}$ from a graph $\mathbb{G}$ such that
the relative $\gamma$-decision $\varepsilon$-superreduct property forces the positive region to cover
more objects than there are vertices, which forces at least one auxiliary object to be in the positive
region. This, in turn, requires the attribute subset to form a dominating set in $\mathbb{G}$.
Minimality of the dominating set translates to minimality of the reduct.

## Proof

We show a polynomial reduction from the Minimal Dominating Set problem to the problem of finding a
minimal relative $\gamma$-decision $\varepsilon$-reduct.

### Construction of the Decision Table

Given an undirected graph $\mathbb{G} = (\mathbb{V}, \mathbb{E})$, construct the decision table
$\mathbb{A}^{\varepsilon}_{\mathbb{G}} = (U^{\varepsilon}_{\mathbb{G}},
A^{\varepsilon}_{\mathbb{G}} \cup \{d^{\varepsilon}_{\mathbb{G}}\})$ as follows.

**Conditional attributes.** One binary attribute per vertex:
$A^{\varepsilon}_{\mathbb{G}} = \{a_1, a_2, \ldots, a_{|\mathbb{V}|}\}$. For object $u_i$
($i = 1, \ldots, |\mathbb{V}|$) and attribute $a_j$ define:

$$
a_j(u_i) =
\begin{cases}
1 & \text{if } i = j \lor (v_i, v_j) \in \mathbb{E}, \\
0 & \text{otherwise}.
\end{cases}
$$

That is, $a_j(u_i) = 1$ when $v_i$ is vertex $v_j$ itself or is adjacent to $v_j$.

**Auxiliary objects.** Let
$t(\varepsilon) = \left\lfloor \frac{|\mathbb{V}|\varepsilon}{1 - \varepsilon} + 1 \right\rfloor$.
Add $t(\varepsilon)$ objects $u_{|\mathbb{V}|+1}, \ldots, u_{|\mathbb{V}|+t(\varepsilon)}$ with all
conditional attributes set to 0: $a_j(u_i) = 0$ for all $i > |\mathbb{V}|$ and all $j$.

**Decision attribute.** Decision classes partition the objects:
$$
d^{\varepsilon}_{\mathbb{G}}(u_i) =
\begin{cases}
0 & \text{if } i \leq |\mathbb{V}|, \\
1 & \text{if } i > |\mathbb{V}|.
\end{cases}
$$

**Consistency.** $\mathbb{A}^{\varepsilon}_{\mathbb{G}}$ is consistent: each $u_i$ with $i \leq
|\mathbb{V}|$ has $a_i(u_i) = 1$, while every auxiliary object has all attributes zero, so
$\gamma(A^{\varepsilon}_{\mathbb{G}}) = 1$.

### The Bound t(epsilon) Ensures Superreduct Coverage of Auxiliary Objects

By construction, $t(\varepsilon) > \frac{|\mathbb{V}|\varepsilon}{1 - \varepsilon}$. The
relative $\gamma$-decision $\varepsilon$-superreduct condition $\gamma(B) \geq (1 - \varepsilon)
\gamma(A^{\varepsilon}_{\mathbb{G}})$ with $\gamma(A^{\varepsilon}_{\mathbb{G}}) = 1$ becomes
$\gamma(B) \geq 1 - \varepsilon$. Expanding:

$$
\frac{|POS(B)|}{|\mathbb{V}| + t(\varepsilon)} \geq 1 - \varepsilon
\implies |POS(B)| \geq (1 - \varepsilon)(|\mathbb{V}| + t(\varepsilon)).
$$

From $t(\varepsilon)(1 - \varepsilon) > |\mathbb{V}|\varepsilon$ we obtain $(t(\varepsilon) +
|\mathbb{V}|)(1 - \varepsilon) > |\mathbb{V}|$, and therefore:

$$
|POS(B)| \geq (1 - \varepsilon)(|\mathbb{V}| + t(\varepsilon)) > |\mathbb{V}|.
$$

Thus any relative $\gamma$-decision $\varepsilon$-superreduct $B$ must place at least one auxiliary
object in $POS(B)$.

### From Superreduct to Dominating Set

Let $B \subseteq A^{\varepsilon}_{\mathbb{G}}$ be a relative $\gamma$-decision
$\varepsilon$-superreduct. Define $\mathbb{W}_B = \{v_j \in \mathbb{V} : a_j \in B\}$.

We claim that every vertex $v_i \in \mathbb{V}$ belongs to $Cov_{\mathbb{G}}(\mathbb{W}_B)$, i.e.,
$\mathbb{W}_B$ is a dominating set for $\mathbb{G}$.

Suppose, for contradiction, that some $v_i \in \mathbb{V}$ is not covered by $\mathbb{W}_B$. Then
$v_i \notin \mathbb{W}_B$ and $v_i$ has no neighbour in $\mathbb{W}_B$. By the definition of the
conditional attributes:

$$
a_j(u_i) = 0 \quad \text{for every } a_j \in B.
$$

Hence $u_i$ is $B$-indiscernible from every auxiliary object $u_k$ (with $k > |\mathbb{V}|$),
since auxiliary objects also have all attributes in $B$ equal to zero. The indiscernibility class
$[u_i]_B$ thus contains:

- $u_i$ (decision class 0), and
- every auxiliary object $u_k$ for $k > |\mathbb{V}|$ (decision class 1).

This class mixes objects from two different decision classes. Therefore **none** of its members
belong to $POS(B)$. In particular, the auxiliary objects $\{u_{|\mathbb{V}|+1}, \ldots,
u_{|\mathbb{V}|+t(\varepsilon)}\}$ are not in $POS(B)$.

But this contradicts the previous section, which proved that $|POS(B)| > |\mathbb{V}|$ and that all
auxiliary objects must lie in $POS(B)$. The contradiction shows that no $v_i$ can be left uncovered
by $\mathbb{W}_B$. Hence:

- For every $v_i \in \mathbb{V}$, either $v_i \in \mathbb{W}_B$ (so $a_i \in B$ and $a_i(u_i) = 1$),
  or $v_i$ is adjacent to some $v_j \in \mathbb{W}_B$ (so $a_j \in B$ and $a_j(u_i) = 1$).

In both cases $v_i \in Cov_{\mathbb{G}}(\mathbb{W}_B)$. Therefore $\mathbb{W}_B$ is a dominating
set for $\mathbb{G}$.

### Minimality Correspondence

Suppose, for contradiction, that $B$ is a minimal relative $\gamma$-decision $\varepsilon$-reduct
but $\mathbb{W}_B$ is not a minimal dominating set. Then there exists a dominating set $\mathbb{W}'
\subsetneq \mathbb{W}_B$ with $|\mathbb{W}'| < |\mathbb{W}_B|$. Let $B' = \{a_j \in
A^{\varepsilon}_{\mathbb{G}} : v_j \in \mathbb{W}'\}$.

Because $\mathbb{W}'$ is a dominating set, every $v_i \in \mathbb{V}$ is either in $\mathbb{W}'$ or
adjacent to a member of $\mathbb{W}'$. In the decision table this means: for each $u_i$ ($i \leq
|\mathbb{V}|$) there exists $a_j \in B'$ with $a_j(u_i) = 1$, so $B'$ discerns all vertex-objects
from the auxiliary objects. Hence $B'$ is also a relative $\gamma$-decision $\varepsilon$-superreduct,
but with $|B'| < |B|$. This contradicts the minimality of $B$. Therefore the minimal
$\gamma$-decision $\varepsilon$-reduct $B$ corresponds to a minimal dominating set $\mathbb{W}_B$.

Thus, if we could find a minimal relative $\gamma$-decision $\varepsilon$-reduct in polynomial
time, we could also solve the Minimal Dominating Set problem in polynomial time, contradicting its
NP-hardness.
