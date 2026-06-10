---
id: prop-equiv-classes-monotonicity
type: proposition
status: complete
created: 2026-06-10
updated: 2026-06-10
tags: [core, indiscernibility]
requires: [concept-decision-table, concept-indiscernibility]
see_also:
  [concept-decision-reduct,
   concept-bireduct-ensemble]
source: "Slezak & Stawicki, 'Complexity of Searching for the Simplest Reduct Matrix Ensembles'
  (paper in preparation)"
---

# Monotonicity of Equivalence Class Count Under Attribute Subset Inclusion

The number of equivalence classes induced by $IND(B)$ is monotonically non-decreasing with respect
to attribute subset inclusion. Adding attributes can only refine or preserve the partition, never
coarsen it.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$. For $B \subseteq B' \subseteq A$:

$$
|U/B| \leq |U/B'|
$$

Moreover, $|U/B| = |U/B'|$ if and only if $U/B = U/B'$.

## Proof

### Part 1: $|U/B| \leq |U/B'|$

Define a function $f: U/B' \to U/B$ by:

$$
f(E_{B'}) = \bigcup_{u \in E_{B'}} [u]_B \quad \text{for } E_{B'} \in U/B'
$$

Since $B \subseteq B'$, any two objects $u, u'$ that are indiscernible under $IND(B')$ are also
indiscernible under $IND(B)$. Therefore, all objects within a single equivalence class $E_{B'} \in
U/B'$ belong to the same equivalence class under $IND(B)$. This means $f(E_{B'}) = [u]_B \in U/B$
for any $u \in E_{B'}$ (such $u$ exists because equivalence classes are non-empty by reflexivity),
so $f$ is well-defined.

To show surjectivity, let $E_B \in U/B$ be arbitrary. Since equivalence classes are non-empty,
choose any $u \in E_B$ (so $E_B = [u]_B$). Then $[u]_{B'} \in U/B'$ and
$f([u]_{B'}) = [u]_B = E_B$. Hence $f$ is a surjection.

A surjection from $U/B'$ onto $U/B$ implies $|U/B| \leq |U/B'|$.

### Part 2: $|U/B| = |U/B'| \Longleftrightarrow U/B = U/B'$

$(\Rightarrow)$ Suppose $|U/B| = |U/B'|$. Since $f$ is a surjection between finite sets of equal
cardinality, $f$ is a bijection. By definition, every $u \in E_{B'}$ satisfies
$u \in f(E_{B'})$, so $E_{B'} \subseteq f(E_{B'})$.

Assume for contradiction that there exists $E_{B'} \in U/B'$ with $E_{B'} \neq f(E_{B'})$, which
means $E_{B'} \subsetneq f(E_{B'})$. Then there exists $u \in f(E_{B'}) \setminus E_{B'}$. This
object $u$ is indiscernible from the elements of $f(E_{B'})$ under $IND(B)$ (since
$f(E_{B'}) \in U/B$), but discernible from the elements of $E_{B'}$ under $IND(B')$ (since
$u \notin E_{B'}$). For its own class $[u]_{B'}$, we have $[u]_{B'} \neq E_{B'}$ but also
$f([u]_{B'}) = f(E_{B'})$ (because $u \in f(E_{B'})$ and $f(E_{B'})$ is a single $B$-class). This
contradicts the injectivity of $f$. Hence $E_{B'} = f(E_{B'})$ for all $E_{B'} \in U/B'$, which
means $f$ is the identity and $U/B = U/B'$.

$(\Leftarrow)$ If $U/B = U/B'$, then trivially $|U/B| = |U/B'|$.

## Consequences

This result is used in the description length framework for bireduct ensembles. Combined with the
identity $|X/B| = |U/B|$ for bireducts, it ensures that adding attributes to a bireduct's attribute
set cannot decrease its description length.
