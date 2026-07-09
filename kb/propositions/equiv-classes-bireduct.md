---
id: prop-equiv-classes-bireduct
type: proposition
status: complete
created: 2026-06-10
updated: 2026-06-11
tags: [core, bireducts, indiscernibility]
requires:
  [concept-decision-table,
   concept-indiscernibility,
   concept-decision-bireduct]
see_also:
  [prop-equiv-classes-monotonicity,
   concept-bireduct-ensemble]
source: "Slezak & Stawicki, 'Complexity of Searching for the Simplest Reduct Matrix Ensembles'
  (paper in preparation)"
---

# Equivalence Class Count for Bireducts Equals Global Count

For a decision bireduct $(X, B)$, the number of equivalence classes induced by $IND_X(B)$ on $X$
equals the number of equivalence classes induced by $IND(B)$ on $U$. This identity connects the
local structure (restricted to bireduct objects) with the global structure (on the full universe).

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ and a decision bireduct $(X, B)$ for $\mathbb{A}$ be given.
Then:

$$
|X/B| = |U/B|
$$

where $X/B$ denotes the quotient set $X/IND_X(B)$ and $U/B$ denotes $U/IND(B)$.

## Proof

Define a function $f: X/B \to U/B$ by:

$$
f(E_X) = \bigcup_{u \in E_X} [u]_B \quad \text{for } E_X \in X/B
$$

### Well-definedness

Since $X \subseteq U$ and equivalence classes are non-empty (by reflexivity), for each $E_X \in
X/B$ we can choose any $u \in E_X$. If $u, u' \in E_X$, then $u$ and $u'$ are indiscernible under
$IND_X(B)$, meaning they have the same values on all attributes from $B$. Since $X \subseteq U$,
they are also indiscernible under $IND(B)$ on $U$. Therefore $[u]_B = [u']_B$, and $f(E_X) =
[u]_B \in U/B$. The function is well-defined.

### Injectivity

Let $E_{X,1}, E_{X,2} \in X/B$ with $u_1 \in E_{X,1}$ and $u_2 \in E_{X,2}$. Then $E_{X,1} =
[u_1]_B^X$ and $E_{X,2} = [u_2]_B^X$. Suppose $f([u_1]_B^X) = f([u_2]_B^X)$, which means
$[u_1]_B = [u_2]_B$. Then $u_1$ and $u_2$ have the same values on all attributes from $B$, so
$[u_1]_B^X = [u_2]_B^X$, i.e., $E_{X,1} = E_{X,2}$. Hence $f$ is injective.

### Surjectivity

Suppose for contradiction that $f$ is not surjective, i.e., there exists $E_U \in U/B \setminus
\text{Im}(f)$, where $\text{Im}(f) = \{f(E_X) : E_X \in X/B\}$. Since equivalence classes are
non-empty, there exists $u \in E_U$.

Neither $u$ nor any object $u'$ indiscernible from $u$ under $IND(B)$ can belong to $X$. If some
$u' \in [u]_B$ were in $X$, then $[u']_B^X \in X/B$ and $f([u']_B^X) = [u']_B = [u]_B = E_U$,
contradicting $E_U \notin \text{Im}(f)$.

Therefore, $X \cup \{u\} \supsetneq X$. Since no object in $[u]_B$ belongs to $X$, adding $u$ to
$X$ preserves the functional dependency, i.e., $B \Rrightarrow_{X \cup \{u\}} d$ still holds. This
contradicts the maximality of $X$ in the decision bireduct $(X, B)$.

Hence $f$ must be surjective.

### Conclusion

Since $f$ is both injective and surjective, it is a bijection. Therefore $|X/B| = |U/B|$.

## Consequences

A direct consequence is that $X$ must contain at least one representative object from each
equivalence class determined by $IND(B)$ on $U$. This follows from the surjectivity of $f$: for
every $E_U \in U/B$, there exists $E_X \in X/B$ such that $f(E_X) = E_U$, meaning some object in
$E_X \subseteq X$ belongs to $E_U$.

This result is fundamental for the description length framework. Combined with
[prop-equiv-classes-monotonicity](../propositions/equiv-classes-monotonicity.md), it ensures that
for a bireduct $(X, B)$, the description length $BirDesc(X, B) = |X/B| \cdot (|B| + 1)$
can be equivalently expressed as $|U/B| \cdot (|B| + 1)$, connecting the bireduct's complexity to
the global partition structure.
