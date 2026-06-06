---
id: prop-decision-bireduct-ordering
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [bireducts, algorithms, ordering]
requires:
  [concept-decision-table,
   concept-decision-bireduct,
   prop-monotony-properties]
see_also:
  [prop-gamma-decision-bireduct-ordering,
   concept-decision-bireduct,
   prop-monotony-properties]
source: tmp/phd/thesis.tex
---

# Correctness of the Decision Bireduct Ordering Algorithm

The ordering algorithm (Algorithm 1: Decision Bireduct Ordering) always produces a valid decision
bireduct, and every decision bireduct is achievable by some choice of input permutation.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. For each permutation $\sigma : \{1, \ldots, |U| +
|A|\} \rightarrow \{1, \ldots, |U| + |A|\}$ the final outcome $(X_{|U|+|A|}, B_{|U|+|A|})$ of the
ordering algorithm is a decision bireduct. Moreover, for each decision bireduct $(X, B)$ there
exists an input $\sigma$ for which the algorithm's output equals $(X, B)$.

## Background

The ordering algorithm starts with $X_0 \leftarrow \emptyset$ and $B_0 \leftarrow A$, then iterates
over $i = 1, \ldots, |U|+|A|$:

- If $\sigma(i) \le |U|$ (an object): add $u_{\sigma(i)}$ to $X_{i-1}$ if the functional dependency
  $B_{i-1} \Rrightarrow_{X_{i-1} \cup \{u_{\sigma(i)}\}} d$ holds.
- If $\sigma(i) > |U|$ (an attribute): remove $a_{\sigma(i)-|U|}$ from $B_{i-1}$ if the functional
  dependency $B_{i-1} \setminus \{a_{\sigma(i)-|U|}\} \Rrightarrow_{X_{i-1}} d$ holds.

## Proof

The proof has two parts: (1) the output is always a decision bireduct, and (2) every decision
bireduct is achievable.

### Part 1: The Output Is a Decision Bireduct

Let a permutation $\sigma$ be given.

**Functional dependency is preserved.** Initially $(X_0, B_0) = (\emptyset, A)$ trivially satisfies
$B_0 \Rrightarrow_{X_0} d$ (the condition is vacuously true over an empty $X$). In each iteration,
an attribute is removed or an object is added only if the dependency remains true. Therefore
$B_{|U|+|A|} \Rrightarrow_{X_{|U|+|A|}} d$ holds at the end.

**Attribute irreducibility.** Suppose for contradiction that there exists $B' \subsetneq
B_{|U|+|A|}$ with $B' \Rrightarrow_{X_{|U|+|A|}} d$. Take $a_j \in B_{|U|+|A|} \setminus B'$.
Let $i$ be the index with $\sigma(i) = j + |U|$ (the iteration where $a_j$ is considered for
removal). At step $i$ we have $B_i \supseteq B_{|U|+|A|} \supsetneq B'$, so $B_i \setminus \{a_j\}
\supseteq B'$, and by the monotony property (Proposition 3), $B_i \setminus \{a_j\}
\Rrightarrow_{X_i} d$. Thus the algorithm would have removed $a_j$ at step $i$, contradicting
$a_j \in B_{|U|+|A|}$.

**Object maximality.** Suppose for contradiction that there exists a proper superset $X' \supsetneq
X_{|U|+|A|}$ with $B_{|U|+|A|} \Rrightarrow_{X'} d$. Take $u_j \in X' \setminus X_{|U|+|A|}$.
Let $i$ be the index with $\sigma(i) = j$ (the iteration where $u_j$ is considered for addition).
At step $i$ we have $X_i \subseteq X_{|U|+|A|} \subsetneq X'$, so $X_i \cup \{u_j\} \subseteq X'$
and by monotony $B_i \Rrightarrow_{X_i \cup \{u_j\}} d$. Thus the algorithm would have added $u_j$
at step $i$, contradicting $u_j \notin X_{|U|+|A|}$.

All three conditions of a decision bireduct are satisfied.

### Part 2: Every Decision Bireduct Is Achievable

Let $(X, B)$ be an arbitrary decision bireduct. Construct a permutation $\sigma$ with the
following four consecutive segments:

1. **$A \setminus B$** (positions $1$ to $|A \setminus B|$): $a_{\sigma(i)-|U|} \in A \setminus B$.
2. **$X$** (positions $|A \setminus B| + 1$ to $|A \setminus B| + |X|$): $u_{\sigma(i)} \in X$.
3. **$B$** (positions $|A \setminus B| + |X| + 1$ to $|X| + |A|$): $a_{\sigma(i)-|U|} \in B$.
4. **$U \setminus X$** (positions $|X| + |A| + 1$ to $|U| + |A|$): $u_{\sigma(i)} \in U \setminus
   X$.

**Effect of segment 1:** The algorithm starts with $X_0 = \emptyset$. For any attribute $a \in A
\setminus B$, removing it does not break the dependency (since $B \subsetneq A$, removal of an
attribute outside $B$ is safe). After $|A \setminus B|$ steps, $B_{|A \setminus B|} = B$.

**Effect of segment 2:** With $B_i = B$ fixed, each object $u \in X$ can be added because
$(X, B)$ is a bireduct -- $B \Rrightarrow_{X} d$, and by monotony adding elements of $X$ one by one
preserves the dependency. After this segment, $X_{|A \setminus B| + |X|} = X$.

**Effect of segments 3 and 4:** Since $(X_{|A \setminus B| + |X|}, B_{|A \setminus B| + |X|}) =
(X, B)$ is already a decision bireduct, no further attributes can be removed and no further objects
can be added. Thus $X_{|U|+|A|} = X$ and $B_{|U|+|A|} = B$.

Therefore $(X, B)$ is exactly the output of the algorithm on permutation $\sigma$.

## Remarks

An important practical observation: the permutation controls which decision classes are represented
within the bireduct. For consistent tables, arranging all objects before all attributes yields
standard decision reducts. For inconsistent tables, the ordering of objects at the beginning of the
permutation decides which decision classes survive within each $A$-induced indiscernibility class.
