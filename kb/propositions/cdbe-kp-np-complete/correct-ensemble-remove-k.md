---
id: prop-correct-ensemble-remove-k
type: proposition
status: complete
created: 2026-07-17
updated: 2026-07-17
tags: [complexity, bireducts, ensemble]
requires:
  - prop-0-1-bireduct-ensemble-decomposition
  - prop-correct-ensemble-m-nonempty
  - prop-set-cover-construction
  - prop-solution-bireduct-properties
  - concept-bireduct-ensemble
see_also:
  - prop-cdbe-kp-np-complete
  - prop-bireduct-replacement
source: "Slezak & Stawicki, 'Complexity of Searching for the Simplest Reduct Matrix Ensembles'
  (paper in preparation)"
---

# Removing $\mathcal{K}$ from a Correct 0-1-Bireduct Ensemble

From any correct 0-1-bireduct ensemble for the transformed table $\mathbb{A}_{\mathcal{S}}$, the
$\mathcal{K}$-multiset (bireducts of the form $(\{u_*\}, \emptyset)$) can be entirely removed while
preserving correctness, possibly reducing the $\mathcal{L}$-multiset as well.

## Statement

Let $(W, \mathcal{S})$ be a Set Cover instance with $\bigcup \mathcal{S} = W$ and
$W \neq \emptyset$, let $\mathbb{A}_{\mathcal{S}}$ be the decision table constructed in
[prop-set-cover-construction](set-cover-construction.md), and let $\mathcal{B}$ be a correct
0-1-bireduct ensemble for $\mathbb{A}_{\mathcal{S}}$. By
[prop-0-1-bireduct-ensemble-decomposition](0-1-bireduct-ensemble-decomposition.md), $\mathcal{B}$
can be uniquely represented as $\mathcal{B} = \mathcal{K} \cup \mathcal{L} \cup \mathcal{M}$.
Then there exists a correct ensemble $\mathcal{B}' \subseteq \mathcal{B}$ such that
$\mathcal{B}' = \mathcal{L}' \cup \mathcal{M}$, where $\mathcal{L}' \subseteq \mathcal{L}$.

## Proof

Assign the cardinalities of the multisets: $|\mathcal{K}| = K$, $|\mathcal{L}| = L$,
$|\mathcal{M}| = \sum_{b \in B} M_b = M$. The following facts follow from the decomposition and
the correctness of $\mathcal{B}$:

- $|\mathcal{B}| = K + L + M$.

- $\forall u \in U_{\mathcal{S}} \cup \{u_*\} \;
  cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}}(u) > |\mathcal{B}|/2 = (K + L + M)/2$.

- Since $\mathcal{K}, \mathcal{L}, \mathcal{M}$ are disjoint,
  $cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}}(u) =
  cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{K}}(u) +
  cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{L}}(u) +
  cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u)$.

- By [prop-solution-bireduct-properties](solution-bireduct-properties.md), $u_*$ is covered by
  all bireducts from $\mathcal{K}$ and $\mathcal{M}$ and none from $\mathcal{L}$:
  $cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{K}}(u_*) = K$,
  $cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u_*) = M$,
  $cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{L}}(u_*) = 0$.

- For any $u \in U_{\mathcal{S}}$, it is covered by all bireducts from $\mathcal{L}$ and none from
  $\mathcal{K}$:
  $cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{K}}(u) = 0$,
  $cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{L}}(u) = L$.

- By [prop-correct-ensemble-m-nonempty](correct-ensemble-m-nonempty.md), $\mathcal{M}$ is
  non-empty, therefore $M > 0$.

### Derived Inequalities

For $u_*$, combining the coverage facts and correctness:

$$
\begin{aligned}
K + M &= cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{K}}(u_*)
      + cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u_*) \\
      &= cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{K}}(u_*)
       + \underbrace{cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{L}}(u_*)}_{=0}
       + cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u_*)
       = cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}}(u_*)
       > \frac{K + L + M}{2},
\end{aligned}
$$

hence $M > L - K$.

For any $u \in U_{\mathcal{S}}$, similarly:

$$
\begin{aligned}
L + cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u)
      &= \underbrace{cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{K}}(u)}_{=0}
       + cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{L}}(u)
       + cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u) \\
      &= cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}}(u)
       > \frac{K + L + M}{2},
\end{aligned}
$$

hence $cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u) > (K - L + M)/2$.

### Case Analysis

There are two options: $|\mathcal{K}| \leq |\mathcal{L}|$ or $|\mathcal{K}| > |\mathcal{L}|$.

**Case 1: $|\mathcal{K}| \leq |\mathcal{L}|$.** Put $\mathcal{B}' = \mathcal{L}' \cup \mathcal{M}$
where $\mathcal{L}' = \{(U_{\mathcal{S}}, \emptyset) \times (L-K)\}$. Clearly
$\mathcal{L}' \subseteq \mathcal{L}$ and $|\mathcal{B}'| = (L - K) + M$.

For $u_*$, using the coverage facts and $\mathcal{L}' \subseteq \mathcal{L}$:

$$
cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}'}(u_*) =
  \underbrace{cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{L}'}(u_*)}_{=0}
  + cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u_*)
  = M = \frac{M}{2} + \frac{M}{2}.
$$

Applying $M > L - K$ to one of the $M/2$ terms gives:

$$
cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}'}(u_*)
  > \frac{L - K}{2} + \frac{M}{2}
  = \frac{L - K + M}{2} = \frac{|\mathcal{B}'|}{2}.
$$

For any $u \in U_{\mathcal{S}}$, since $u$ is covered by all bireducts from
$\mathcal{L}' = \{(U_{\mathcal{S}}, \emptyset) \times (L-K)\}$:

$$
\begin{aligned}
cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}'}(u)
  &= cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{L}'}(u)
     + cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u) \\
  &= (L - K) + cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u).
\end{aligned}
$$

Applying $cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u) > (K - L + M)/2$ yields:

$$
cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}'}(u)
  > L - K + \frac{K - L + M}{2}
  = \frac{L - K + M}{2} = \frac{|\mathcal{B}'|}{2}.
$$

Thus $\mathcal{B}'$ is correct.

**Case 2: $|\mathcal{K}| > |\mathcal{L}|$.** Put $\mathcal{B}' = \mathcal{L}' \cup \mathcal{M}$
where $\mathcal{L}' = \emptyset$. Clearly $\emptyset \subseteq \mathcal{L}$ and
$|\mathcal{B}'| = M$.

For $u_*$, using the coverage facts and $\mathcal{L}' = \emptyset$:

$$
cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}'}(u_*) =
  \underbrace{cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{L}'}(u_*)}_{=0}
  + cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u_*)
  = M.
$$

Since $M > 0$, we have $cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}'}(u_*) > M/2 = |\mathcal{B}'|/2$.

For any $u \in U_{\mathcal{S}}$, since $\mathcal{L}' = \emptyset$:

$$
cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}'}(u) =
  \underbrace{cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{L}'}(u)}_{=0}
  + cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u).
$$

Applying $cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{M}}(u) > (K - L + M)/2$ and using $K > L$:

$$
\begin{aligned}
cov_{\mathbb{A}_{\mathcal{S}}, \mathcal{B}'}(u)
  &> \frac{K - L + M}{2}
   = \underbrace{\frac{K - L}{2}}_{>0} + \frac{M}{2}
   > \frac{M}{2} = \frac{|\mathcal{B}'|}{2}.
\end{aligned}
$$

Thus $\mathcal{B}'$ is correct in this case as well.

### Conclusion

In both cases $K \leq L$ and $K > L$, a correct ensemble $\mathcal{B}' \subseteq \mathcal{B}$
exists with $\mathcal{B}' = \mathcal{L}' \cup \mathcal{M}$ and
$\mathcal{L}' \subseteq \mathcal{L}$. ∎

## Consequences

This lemma establishes that the $\mathcal{K}$-multiset (bireducts covering only $u_*$) is never
essential for correctness: it can always be removed, possibly with a corresponding reduction of
the $\mathcal{L}$-multiset. In the CDBEkP NP-completeness reduction, this means that attention can
be restricted to ensembles consisting only of $\mathcal{L}$-type and $\mathcal{M}$-type bireducts,
which correspond more directly to the Set Cover structure.
