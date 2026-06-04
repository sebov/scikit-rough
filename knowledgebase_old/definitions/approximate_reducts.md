---
tags: [rst, core, reduction]
related: [notation_and_symbols.md, definitions/decision_table.md, definitions/positive_region.md, definitions/reducts.md, definitions/decision_rules.md, propositions/approximate_reduct_complexity.md]
---

# Approximate Decision Reducts

The concept of a decision reduct is fundamental to rough set theory. However, practical applications
often involve large and noisy data sets. To address such challenges, several generalizations have
been proposed, such as dynamic reducts and approximate decision reducts. An approximate decision
reduct is an irreducible subset of attributes that, under a specified criterion, retains
decision-related information above a given threshold.

Criteria for approximate decision reducts are based on monotone evaluation functions
$F : 2^A \rightarrow [0, 1]$ that measure the degree of decision information induced by a subset
of attributes. Two variants are distinguished in the literature.

## Relative Approximate Decision Reduct

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. Let $\varepsilon \in [0, 1)$ and a
nondecreasing monotone (with respect to set inclusion) function $F : 2^A \rightarrow [0, 1]$ be
given. A subset $B \subseteq A$ is a relative $\varepsilon$-$F$-**superreduct** if and only if:

$$
  F(B) \geq (1 - \varepsilon)F(A)
$$

A subset $B \subseteq A$ is a relative $\varepsilon$-$F$-**reduct** if and only if it satisfies the
above inequality and none of its proper subsets does.

## Approximate Decision Reduct

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. Let $\varepsilon \in [0, 1)$ and a
nondecreasing monotone (with respect to set inclusion) function $F : 2^A \rightarrow [0, 1]$ be
given. A subset $B \subseteq A$ is an $\varepsilon$-$F$-**superreduct** if and only if:

$$
  F(B) \geq 1 - \varepsilon
$$

A subset $B \subseteq A$ is an $\varepsilon$-$F$-**reduct** if and only if it satisfies the above
inequality and none of its proper subsets does.

The relative version evaluates subsets in relation to the full set of attributes (threshold
$(1-\varepsilon)F(A)$), while the absolute version uses a fixed threshold $(1-\varepsilon)$.

## Example Evaluation Functions

The function $\gamma$ (dependency degree) defined in `positive_region.md` is one example of $F$.
Additional examples are given below.

### Majority Function $M$

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. Let $\{D_1, \dots, D_{|V_d|}\}$
denote the partition of $U$ induced by the decision attribute $d$, i.e., $D_k = \{u \in U : d(u) = k\}$.
The majority function $M : 2^A \rightarrow [0, 1]$ is defined for $B \subseteq A$ as:

$$
  M(B) = \frac{1}{|U|}
    \sum_{[u]_B \in U/B}
    \max_{k = 1, \dots, |V_d|}
    |[u]_B \cap D_k|
$$

$M(B)$ measures the accuracy of a rule-based classifier that, for each $B$-induced equivalence
class, points at the most frequent decision within that class.

### Relative Gain Function $R$

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. The relative gain function
$R : 2^A \rightarrow [0, 1]$ is defined for $B \subseteq A$ as:

$$
  R(B) = \frac{1}{|V_d|}
    \sum_{[u]_B \in U/B}
    \max_{k = 1, \dots, |V_d|}
    \frac{|[u]_B \cap D_k|}{|D_k|}
$$

$R(B)$ extends the classical rough set model with a Bayesian perspective. For each $B$-induced
equivalence class, it considers the decision class that becomes maximally frequent compared to its
overall occurrence in the data, then averages over decision classes.

## Relationship to Other Reduct Types

For a consistent decision table $\mathbb{A}$, we have $\gamma(A) = M(A) = R(A) = 1$. Consequently,
for consistent tables the relative and absolute $\varepsilon$-$F$-reducts are equivalent whenever
$F$ is $\gamma$, $M$, or $R$. Moreover, for consistent tables, relative
$0$-$\{\gamma, M, R\}$-reducts are all equivalent to standard decision reducts.

For inconsistent tables, these notions serve as alternative ways to extend decision reducts, and may
lead to different subsets of attributes for the same data.

## Role of the Threshold $\varepsilon$

For $\varepsilon$-$F$-reducts, the value of $\varepsilon$ can be understood as a threshold for the
allowed decrease of classifier accuracy and can address the balance between simplicity and
confidence of rules.

- **Higher $\varepsilon$**: $\varepsilon$-$F$-reducts usually contain fewer attributes and the
  generated decision rules become shorter. By accepting slight inconsistencies we gain higher
  simplicity and a higher chance that attribute values observed for previously unseen cases will
  match predecessors of some existing rules.
- **Lower $\varepsilon$**: $\varepsilon$-$F$-reducts contain more attributes and rules generated
  based on those attributes are potentially more complex but also more accurate over the decision
  table treated as training data.

## Complexity

The search for minimal $\varepsilon$-$F$-reducts is NP-hard for functions such as $M$ and $R$. The
standard proof technique involves a polynomial reduction from the Minimal Dominating Set Problem,
a classic NP-hard problem.

### Graph

A graph is an ordered pair $G = (V, E)$, where $V$ is the set of vertices (or nodes) and
$E \subseteq V \times V$ is the set of edges (or links). A graph is **non-directed** if the
relation $E$ is symmetric.

### Dominating Set

Let a non-directed graph $G = (V, E)$ be given. A subset $W \subseteq V$ is a **dominating set**
for $G$ if and only if:

$$
  Cov_G(W) = V
$$

where $Cov_G(W) = W \cup \{ v \in V : \exists_{w \in W} \; (v, w) \in E \}$ is the set of vertices
that either belong to $W$ or are adjacent to at least one member of $W$.

### Minimal Dominating Set Problem

The Minimal Dominating Set Problem is an optimization problem of finding a minimal (by cardinality)
dominating set for a given undirected graph $G = (V, E)$. This problem is **NP-hard**.
