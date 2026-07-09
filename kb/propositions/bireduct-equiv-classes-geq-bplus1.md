---
id: prop-bireduct-equiv-classes-geq-bplus1
type: proposition
status: complete
created: 2026-07-07
updated: 2026-07-07
tags: [bireducts, complexity]
requires:
  [concept-decision-table,
   concept-indiscernibility,
   concept-decision-bireduct]
see_also:
  [prop-bireduct-desc-len-formula,
   prop-bireduct-desc-len-geq-bplus1-squared,
   prop-equiv-classes-bireduct]
source: "Slezak & Stawicki, 'Complexity of Searching for the Simplest Reduct Matrix Ensembles'
  (paper in preparation)"
---

# Bireduct Equivalence Classes Lower Bound

For any decision bireduct $(X, B)$, the number of equivalence classes determined by
$IND_X(B)$ on $X$ is at least $|B| + 1$. The proof constructs a forest graph over the equivalence
classes and uses the graph-theoretic inequality $|E| \leq |V| - 1$.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be a decision table and $(X, B)$ be a decision bireduct for
$\mathbb{A}$. Then:

$$
|X/B| \geq |B| + 1
$$

where $X/B$ denotes the quotient set $X / IND_X(B)$.

## Proof

Since $(X, B)$ is a decision bireduct, $B$ is irreducible: removing any attribute $b \in B$ would
break the functional dependency $B \Rrightarrow_X d$. Hence, for each $b \in B$, there exists at
least one pair of objects $u, v \in X$ with different decisions such that they differ only on $b$,
i.e., they agree on all attributes in $B \setminus \{b\}$ but differ on $b$.

Since $u$ and $v$ differ on $b$, they belong to distinct equivalence classes $E_b, F_b \in X/B$.
These classes differ on $b$ and agree on all other attributes from $B \setminus \{b\}$. For each
$b \in B$ there may exist multiple such pairs; we arbitrarily pick one pair per $b$.

Construct an undirected graph $G = (V, E)$, where vertices are the equivalence classes
$V = X/B$ and each edge $\{E_b, F_b\} \in E$ corresponds to an attribute $b \in B$ and its chosen
pair. Thus $|V| = |X/B|$, $|E| = |B|$, and each edge is labeled by a distinct attribute.

We show that $G$ is a forest (an acyclic undirected graph). Suppose, by contradiction, that a cycle
exists. Each edge labeled by $b$ connects two classes differing only on $b$; traversing such an
edge changes exactly the value at position $b$ in the vector representing the equivalence class.
After completing the cycle, the starting vertex is reached again, so every attribute value must
equal its original value. Therefore, any attribute $b$ whose edge appears in the cycle must be
traversed at least twice --- otherwise its value would remain different from the original. But each
$b \in B$ labels exactly one edge in $G$, and a simple cycle cannot traverse the same edge twice.
This is a contradiction. Hence $G$ is a forest.

For any forest graph, $|E| \leq |V| - 1$. Substituting $|E| = |B|$ and $|V| = |X/B|$, we obtain
$|B| \leq |X/B| - 1$, and finally:

$$
|X/B| \geq |B| + 1
$$

## Remarks

This result is general: it holds for any decision bireduct over any decision table, regardless of
attribute domains (not only binary attributes). The graph-based proof is self-contained and does
not rely on any intermediate lemmas beyond the definition of a decision bireduct.

Combined with [prop-bireduct-desc-len-formula](../propositions/bireduct-desc-len-formula.md), this
immediately yields the quadratic lower bound
$BirDesc(X, B) \geq (|B| + 1)^2$ (see
[prop-bireduct-desc-len-geq-bplus1-squared](../propositions/bireduct-desc-len-geq-bplus1-squared.md)).
