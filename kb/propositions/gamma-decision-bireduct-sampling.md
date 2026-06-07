---
id: prop-gamma-decision-bireduct-sampling
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [bireducts, algorithms, sampling, positive-region]
requires:
  [concept-decision-table,
   concept-gamma-decision-bireduct,
   concept-gamma-decision-reduct]
see_also:
  [prop-decision-bireduct-sampling,
   concept-gamma-decision-bireduct,
   concept-positive-region]
source: src-thesis-phd
---

# Correctness of the Gamma Decision Bireduct Sampling Algorithm

The gamma-decision bireduct sampling algorithm always produces a valid $\gamma$-decision bireduct,
and every $\gamma$-decision bireduct is achievable. The proof is a simpler version of the ordinary
sampling case.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ and $A^\diamond \subseteq A$ be given. Then the outcome
$(X, B)$ of the $\gamma$-decision bireduct sampling algorithm is a $\gamma$-decision bireduct for
$\mathbb{A}$. Moreover, each $\gamma$-decision bireduct for $\mathbb{A}$ can be obtained as a
result of the algorithm.

## Background

The $\gamma$-sampling algorithm differs from the standard sampling algorithm in two ways:

1. The object set $X$ is not built by collecting all objects sharing the representative's decision.
   Instead, after sampling representatives $U^\diamond$, a $\gamma$-modified decision attribute
   $d_{A^\diamond}^\gamma$ is defined (assigning $\circledast$ to non-positive-region objects).
2. The final output is always $(POS_{\mathbb{A}}(B), B)$ -- the full positive region together with
   the reduct.

## Proof

The proof is a simplified version of the proof of `prop-decision-bireduct-sampling`.

For a given $A^\diamond \subseteq A$, the algorithm constructs a compact version of the
$\gamma$-related table: one representative object per $A^\diamond$-indiscernibility class, with
the decision value assigned according to the $\gamma$-modified decision attribute (using
$\circledast$ for objects not in the positive region of $A^\diamond$). A decision reduct
$B \subseteq A^\diamond$ is then computed for this table. By the properties of
$\gamma$-decision reducts, the resulting bireduct has $X = POS_{\mathbb{A}}(B)$.

For the reverse direction, any $\gamma$-decision bireduct $(X, B)$ with $X = POS_{\mathbb{A}}(B)$
can be obtained by setting $A^\diamond = B$ and selecting representative objects from
$POS_{\mathbb{A}}(B)$.

An important property distinguishing the $\gamma$-case: the selection of representative objects
has no impact on the final result. Regardless of which objects are chosen from each
indiscernibility class, the reduced decision table (with $\gamma$-modified decisions) is the same
up to object numbering.
