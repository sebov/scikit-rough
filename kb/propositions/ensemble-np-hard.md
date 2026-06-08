---
id: prop-ensemble-np-hard
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [complexity, bireducts, ensembles]
requires:
  [concept-bireduct-ensemble,
   concept-decision-bireduct,
   concept-np-hardness-foundations]
see_also:
  [concept-bireduct-ensemble,
   prop-minimal-epsilon-bireduct-np-hard,
   concept-np-hardness-foundations]
source: src-thesis-phd
---

# NP-Hardness of Simplest Correct Decision Bireduct Ensemble Problem

The problem of finding the simplest correct ensemble of decision bireducts (SCDBEP) is NP-hard. The
proof is by polynomial reduction from the minimal dominating set problem.

## Statement

The Simplest Correct Decision Bireduct Ensemble Problem (SCDBEP) is NP-hard.

## Proof Strategy

The goal is a polynomial reduction from the **minimal dominating set problem** to SCDBEP.
We follow the standard three-step structure:

1. **Transformation.** Map every graph $\mathbb{G}$ to a decision table
   $\mathbb{A}_{\mathbb{G}}$ in polynomial time.
2. **(⇒) Feasibility.** If $\mathbb{G}$ has a dominating set of size $n$, then
   $\mathbb{A}_{\mathbb{G}}$ admits a correct ensemble with $n$ bireducts having non-empty
   attribute sets. This shows $OPT_Y \le OPT_X$ (the ensemble cannot be worse than the
   dominating set). The construction is generic: for **any** dominating set of size $n$, we
   produce an ensemble with exactly $n$ single-attribute bireducts and $n-1$ dummy (empty)
   bireducts. The cost under $\prec$ is $n$ (the number of ones in the sorted cardinality
   sequence).
3. **(⇐) Optimality.** Any correct ensemble for $\mathbb{A}_{\mathbb{G}}$ must have at least
   $n$ bireducts with non-empty attributes, where $n$ is the size of a **minimal** dominating
   set in $\mathbb{G}$. This shows $OPT_Y \ge OPT_X$ (the ensemble cannot be better than the
   dominating set). Crucially, this direction only needs to handle the special instances
   $\mathbb{A}_{\mathbb{G}}$ produced by the transformation, not arbitrary decision tables.

Steps (⇒) and (⇐) together yield $OPT_X = OPT_Y$: the cost function is the identity $n \mapsto n$.
Since finding a minimum dominating set is NP-hard, SCDBEP is NP-hard.

Each bireduct in the constructed ensemble is verified to be a genuine decision bireduct (not
just a pair satisfying the functional dependency).

## Proof

We show NP-hardness of SCDBEP by polynomial reduction of the minimal dominating set problem.

### Graph-to-Table Encoding

Let us consider an undirected graph $\mathbb{G} = (\mathbb{V}, \mathbb{E})$ and create a decision
table $\mathbb{A}_{\mathbb{G}} = (U_{\mathbb{G}} \cup \{u_*\}, A_{\mathbb{G}} \cup \{d_*\})$, where:

- For each vertex $v \in \mathbb{V}$, there is an attribute $a_v \in A_{\mathbb{G}}$.
- For each vertex $v' \in \mathbb{V}$, there is an object $u_{v'} \in U_{\mathbb{G}}$.
- There is a special object $u_* \notin U_{\mathbb{G}}$.
- Attribute values: $a_v(u_{v'}) = 1$ if and only if $v = v'$ or $(v, v') \in \mathbb{E}$ (i.e., $v$
  dominates $v'$), and $a_v(u_*) = 0$ for all $v$.
- Decision values: $d(u_{v'}) = 0$ for all $v' \in \mathbb{V}$, and $d(u_*) = 1$.

### Dominating Set Corresponds to Bireduct

$B \subseteq \mathbb{V}$ is a **minimal** dominating set in $\mathbb{G}$ if and only if
$(U_{\mathbb{G}}, B_{\mathbb{G}})$ is a decision bireduct, where $B_{\mathbb{G}} = \{a_v : v \in B\}$.

**Why:** $B$ is a dominating set iff every vertex $v' \in \mathbb{V}$ is either in $B$ or adjacent to
some vertex in $B$. In the decision table, this means every object $u_{v'}$ is discerned from $u_*$
by some attribute in $B_{\mathbb{G}}$ (since $a_v(u_{v'}) = 1$ but $a_v(u_*) = 0$). Thus,
$B_{\mathbb{G}} \Rrightarrow_{U_{\mathbb{G}}} d$ (determination).

**Attribute irreducibility:** If $B$ is minimal, then no proper subset $B' \subsetneq B$ is a
dominating set. This means there exists $v'$ not dominated by any $v \in B'$, so $u_{v'}$ is not
discerned from $u_*$ by any attribute in $B'_{\mathbb{G}}$. Thus, $B'_{\mathbb{G}} \not\Rrightarrow_{U_{\mathbb{G}}} d$,
so $B_{\mathbb{G}}$ is irreducible.

**Object maximality:** Trivially satisfied since $U_{\mathbb{G}}$ is the entire table.

### Single-Element Ensemble is Correct but Not Simplest

It is obvious that a single-element bireduct ensemble $\{(U_{\mathbb{G}}, B_{\mathbb{G}})\}$ is
correct according to the definition of correct ensemble (every object is covered by at least one
bireduct, which is $> 1/2$).

However, we can always construct a simpler (or equally simple if $B_{\mathbb{G}}$ is already a
singleton) correct ensemble.

### Constructing a Simpler Ensemble

Assuming that $B_{\mathbb{G}} = \{a_{v_1}, a_{v_2}, \ldots, a_{v_n}\}$, define:

- Attribute subsets: $B_{\mathbb{G},1} = \{a_{v_1}\}$, ..., $B_{\mathbb{G},n} = \{a_{v_n}\}$,
  $B_{\mathbb{G},n+1} = \emptyset$, ..., $B_{\mathbb{G},2n-1} = \emptyset$.
- Object subsets: $X_{\mathbb{G},1} = \{u_*\} \cup \{u \in U_{\mathbb{G}} : a_{v_1}(u) = 1\}$, ...,
  $X_{\mathbb{G},n} = \{u_*\} \cup \{u \in U_{\mathbb{G}} : a_{v_n}(u) = 1\}$,
  $X_{\mathbb{G},n+1} = U_{\mathbb{G}}$, ..., $X_{\mathbb{G},2n-1} = U_{\mathbb{G}}$.

Then the ensemble $\{(X_{\mathbb{G},1}, B_{\mathbb{G},1}), \ldots, (X_{\mathbb{G},2n-1}, B_{\mathbb{G},2n-1})\}$
is correct according to the definition of correct ensemble.

**Why correct:**
- Objects in $U_{\mathbb{G}}$: each $u_{v'}$ is covered by bireducts $(X_{\mathbb{G},i}, B_{\mathbb{G},i})$
  where $a_{v_i}(u_{v'}) = 1$ (i.e., $v_i$ dominates $v'$). Since $B$ is a dominating set, every
  $v'$ is dominated by at least one $v_i$, so $u_{v'}$ is covered by at least one bireduct with
  non-empty attribute set. Additionally, $u_{v'}$ is covered by all $n-1$ bireducts with empty
  attribute sets (since $X_{\mathbb{G},n+1} = \ldots = X_{\mathbb{G},2n-1} = U_{\mathbb{G}}$).
  Thus, $u_{v'}$ is covered by at least $1 + (n-1) = n$ bireducts out of $2n-1$. Since
  $n > (2n-1)/2$ (equivalently, $2n > 2n-1$), the majority condition is satisfied.
- Special object $u_*$: covered by all bireducts $(X_{\mathbb{G},i}, B_{\mathbb{G},i})$ where
  $u_* \in X_{\mathbb{G},i}$, which is true for $i = 1, \ldots, n$ (by construction). Thus, $u_*$
  is covered by $n$ bireducts out of $2n-1$, which is $> (2n-1)/2$.

### Simplicity Corresponds to Dominating Set Size

The constructed ensemble has attribute cardinalities $(1, 1, \ldots, 1, 0, 0, \ldots, 0)$ (with $n$
ones and $n-1$ zeros). According to the simplicity order $\prec$, this is simpler than any ensemble
with a larger maximum cardinality.

**Why is this the simplest possible?** We must show two things: (a) no correct ensemble can have
fewer than $n$ bireducts with non-empty attribute sets, and (b) no correct ensemble with exactly $n$
bireducts with non-empty attribute sets can have fewer than $n-1$ bireducts with empty attribute
sets.

Suppose an ensemble has $m$ bireducts with single attributes (from $B$) and $k$ bireducts with empty
attributes. For the ensemble to be correct:
- $u_*$ is covered by the $m$ bireducts with non-empty attributes, so $m > (m+k)/2$, i.e., $m > k$.
- If $m < n$, then since $B$ is a **minimal** dominating set, the $m$ chosen vertices do not form a
  dominating set. Thus, there exists $v'$ not dominated by any chosen vertex, so $u_{v'}$ is covered
  by 0 bireducts with non-empty attributes.
- For $u_{v'}$ to satisfy the majority condition, it must be covered by $> (m+k)/2$ bireducts with
  empty attributes, i.e., $k > (m+k)/2$, i.e., $k > m$.

This contradicts $m > k$. Therefore, $m \geq n$, and since $B$ is minimal, we must use exactly $m = n$
(all vertices in $B$). The constructed ensemble with $n$ ones is therefore the simplest possible.

**Why $n-1$ zeros is the minimum.** For $n$ ones and $k$ zeros to form a correct ensemble, consider
any vertex $v_i \in B$. Since $B$ is minimal, removing $v_i$ breaks the dominating set, so there
exists at least one vertex $v'$ that is dominated by $v_i$ but not by any other vertex in $B$. The
corresponding object $u_{v'}$ is covered by $1 + k$ bireducts, so $1 + k > (n + k)/2$, which gives
$k > n - 2$, i.e., $k \ge n - 1$. More zeros would make the ensemble more complex under $\prec$
without benefit -- when comparing two ensembles with the same number of ones, the one with fewer
zeros wins because its $-1$ sentinel appears earlier in the sorted sequence.

Therefore, the simplest correct ensemble corresponds to the smallest dominating set.

### Conclusion

The simplest correct ensemble of decision bireducts corresponds to the smallest dominating set in
the graph $\mathbb{G}$. Since finding the minimal dominating set is NP-hard, SCDBEP is NP-hard.

## Remarks

This proof shows that ensemble optimization is computationally intractable, even for the simplified
notion of simplicity based on maximum attribute cardinality.

The construction also illustrates an interesting property: bireducts with empty attribute sets act as
"dummy" classifiers that always predict the same decision (in this case, $d = 0$). These dummy
classifiers help tune the majority voting mechanism in the ensemble.

The NP-hardness result motivates the use of heuristic search methods for constructing bireduct
ensembles in practice. The ordering-based algorithms discussed in the thesis provide practical
approaches, though they do not guarantee optimality.
