---
id: prop-cdbe-kp-np-membership
type: proposition
status: complete
created: 2026-08-02
updated: 2026-08-02
tags: [complexity, bireducts, ensemble]
requires:
  - concept-decision-table
  - concept-decision-bireduct
  - concept-bireduct-ensemble
  - prop-bireduct-desc-len-formula
see_also:
  - prop-cdbe-kp-np-complete
source: src-reduct-matrix-ensembles
---

# NP Membership of CDBEkP

The Correct Decision Bireduct Ensemble of Size $k$ Problem (CDBEkP) is in NP. A certificate
consisting of an ensemble of bireducts can be verified in polynomial time by a deterministic
Turing machine.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be a decision table and $k \geq 0$ be an integer. The
Correct Decision Bireduct Ensemble of Size $k$ Problem (CDBEkP) -- deciding if there exists a
correct ensemble of bireducts with description length at most $k$ -- is in the NP class. A
certificate $\mathcal{B} = \{(X_1, B_1), \ldots, (X_m, B_m)\}$ consisting of a collection of
object-attribute pairs can be verified in polynomial time by a deterministic Turing machine.

## Proof

We need to verify in polynomial time whether $\mathcal{B}$ is a correct ensemble and its
description length $EnsDesc(\mathcal{B})$ is less than or equal to $k$. Below we present the
verification steps together with their complexity bounds. For the bounds we use inequalities
$|X_i| \leq |U|$ and $|B_i| \leq |A|$. Hence, $O(|X_i|) = O(|U|)$ and $O(|B_i|) = O(|A|)$,
respectively.

1. (cf. [decision bireduct definition](../../concepts/decision-bireduct.md)) Check that each
   element $(X_i, B_i) \in \mathcal{B}$ is a bireduct:

   1. There is $B_i \Rrightarrow_{X_i} d$.
      It is enough to iterate over all pairs in $X_i$ and check whether there are any
      conflicts (same values on all attributes but different decision). Overall,
      $O(|U|^2 \cdot |A|)$.

   2. There is no proper subset $B' \subsetneq B_i$ such that $B' \Rrightarrow_{X_i} d$.
      It is enough to temporarily remove each attribute $b \in B_i$ and check for
      conflicts (same values on all attributes but different decision) between objects
      in $X_i$ -- for each $b \in B_i$, we need to find at least one conflicting pair
      that matches on $B_i \setminus \{b\}$ and differs on $d$ -- confirming that $b$ is
      irreducible. Overall, $O(|A| \cdot |U|^2 \cdot |A|) = O(|A|^2 \cdot |U|^2)$.

   3. There is no proper superset $X' \supsetneq X_i$ such that
      $B_i \Rrightarrow_{X'} d$.
      It is enough to iterate over all objects outside $X_i$ and verify whether each
      such object conflicts with some object already in $X_i$. Overall,
      $O(|U \setminus X_i| \cdot |U| \cdot |A|) = O(|U|^2 \cdot |A|)$.

2. (cf. [correct ensemble definition](../../concepts/bireduct-ensemble.md)) Check that
   $\mathcal{B} = \{(X_1, B_1), \ldots, (X_m, B_m)\}$ is a correct ensemble.
   It is enough to iterate over all objects in $U$ and compute the number of decision
   bireducts that cover each object. Even with a naive scan through each bireduct's object
   set $X_i$, we can obtain $O(|U| \cdot |\mathcal{B}| \cdot |U| \cdot |A|) = O(m \cdot
   |U|^2 \cdot |A|)$.

3. Compute $EnsDesc(\mathcal{B})$ and compare the parameter $k$.
   It is enough to iterate over all bireducts from $\mathcal{B}$, compute description
   length for each of them $BirDesc(X_i, B_i)$ (cf. [the description length
   formula](../bireduct-desc-len-formula.md)), and compare the sum with $k$. The
   equivalence classes can be computed in polynomial time by lexicographically sorting the
   rows based on conditional attributes and counting the distinct groups in a single linear
   scan. Overall, $O(|\mathcal{B}| \cdot (|U| \cdot \log |U| \cdot |A| + |U| \cdot
   |A|)) = O(m \cdot |U| \cdot \log |U| \cdot |A|)$.

All the above checks can be performed in time polynomial in the input size $|U|$, $|A|$, and
$m = |\mathcal{B}|$. Hence, CDBEkP is in NP. ∎
