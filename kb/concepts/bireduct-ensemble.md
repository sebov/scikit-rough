---
id: concept-bireduct-ensemble
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-08
tags: [core, bireducts, evaluation]
requires:
  [concept-decision-table,
   concept-decision-bireduct,
   concept-epsilon-decision-bireduct]
see_also:
  [concept-np-hardness-foundations,
   prop-ensemble-np-hard]
source: src-thesis-phd
---

# Bireduct Ensemble

An ensemble of decision bireducts is a multiset of bireducts that collectively vote on the decision
value of each object. The covered object sets provide explicit control over which classifiers are
correct for which objects.

## Ensemble

Let $\mathbb{A} = (U, A \cup \{d\})$. A multiset
$\mathcal{B} = \{(X_1, B_1), \ldots, (X_m, B_m)\}$, where $m \geq 0$ and each $(X_i, B_i)$ is a
decision bireduct for $\mathbb{A}$, is an **ensemble of decision bireducts** for $\mathbb{A}$. If
$m = 0$, the ensemble is **empty**.

## Coverage Count Function

For an ensemble $\mathcal{B} = \{(X_1, B_1), \ldots, (X_m, B_m)\}$ for $\mathbb{A}$, the **coverage
count function** $cov_{\mathbb{A},\mathcal{B}} : U \to \mathbb{N}$ determines how many bireducts
from $\mathcal{B}$ cover a given object:

$$
cov_{\mathbb{A},\mathcal{B}}(u) = |\{(X_i, B_i) \in \mathcal{B} : u \in X_i\}|
$$

## Correct Ensemble

An ensemble $\mathcal{B} = \{(X_1, B_1), \ldots, (X_m, B_m)\}$ for $\mathbb{A}$ is **correct** if
and only if every object is covered by more than half of the ensemble components:

$$
\forall_{u \in U}\; cov_{\mathbb{A},\mathcal{B}}(u) > \frac{|\mathcal{B}|}{2}
$$

This ensures that a simple majority vote among the $m$ rule-based classifiers is always correct on
the training data.

## Simpler Ensemble

Let $\mathcal{B} = \{(X_1, B_1), \ldots, (X_m, B_m)\}$ and
$\mathcal{C} = \{(Y_1, C_1), \ldots, (Y_n, C_n)\}$ be two correct ensembles. $\mathcal{B}$ is
**simpler** than $\mathcal{C}$, denoted $\mathcal{B} \prec \mathcal{C}$, if the following procedure
determines so:

1. Sort the cardinalities $\lvert B_i \rvert$ and $\lvert C_j \rvert$ in descending order.
2. Append $-1$ to each sequence.
3. Find the first position where the sorted sequences differ.
4. $\mathcal{B} \prec \mathcal{C}$ if $\mathcal{B}$'s value is lower at that position.

This induces a linear order favoring ensembles whose largest attribute subsets are smaller --
analogous to a lexicographic order over sorted cardinality sequences.

## Description Length

An alternative measure of ensemble simplicity that accounts for both attribute and object components
of each bireduct. This measure is based on the total number of descriptors used in decision rules.

For a decision bireduct $(X, B)$ on $\mathbb{A} = (U, A \cup \{d\})$, its **description length**,
denoted $BireductDescLen(X, B)$, is the total number of descriptors (expressions of the form
$a = v$) used in the set of decision rules induced from the bireduct:

$$
BireductDescLen(X, B) = |X/B| \cdot (|B| + 1)
$$

Each rule has $|B| + 1$ descriptors (one per attribute in $B$, plus one for the decision). Objects
in the same equivalence class $E \in X/B$ share a single rule, hence the factor $|X/B|$.

For an ensemble $\mathcal{B} = \{(X_1, B_1), \ldots, (X_m, B_m)\}$ for $\mathbb{A}$, the
**description length of the ensemble**, denoted $EnsembleDescLen(\mathcal{B})$, is the sum of
description lengths of all its members:

$$
EnsembleDescLen(\mathcal{B}) = \sum_{(X_i, B_i) \in \mathcal{B}} BireductDescLen(X_i, B_i)
$$

*Source: Ślęzak & Stawicki, "Complexity of Searching for the Simplest Reduct Matrix Ensembles"
(paper in preparation).*

## Simplest Correct Ensemble Problem (SCDBEP)

The Simplest Correct Decision Bireduct Ensemble Problem is: for an input $\mathbb{A}$, find a correct
ensemble $\mathcal{B}$ such that no other correct ensemble is simpler under $\prec$.

SCDBEP is NP-hard. The proof reduces from the Minimal Dominating Set problem by encoding a graph
$\mathbb{G}$ into a decision table $\mathbb{A}_{\mathbb{G}}$ where the smallest dominating set
corresponds to the simplest correct ensemble. See
[prop-ensemble-np-hard](../propositions/ensemble-np-hard.md) for the full statement and proof.

## Remarks

The ensemble framework gives decision bireducts an advantage over approximate reducts: explicit
knowledge of covered objects allows verifying that different components do not repeat classification
mistakes on the same training objects. This enables diversification strategies analogous to boosting
and bagging but with direct, interpretable control over object coverage.

Even "dummy" classifiers with $B = \emptyset$ (always predicting the majority class) can be valid
ensemble components that help tune the majority voting mechanism.
