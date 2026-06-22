---
id: prop-transformed-table-consistent
type: proposition
status: draft
created: 2026-06-21
updated: 2026-06-21
tags: [complexity, bireducts, ensemble, consistency]
requires:
  - prop-set-cover-construction
  - concept-consistency
see_also:
  - prop-cdbe-kp-np-complete
  - prop-correct-ensemble-iff-dectab-consistent
source: src-reduct-matrix-ensembles
---

# Consistency of the Transformed Decision Table

The decision table $\mathbb{A}_{\mathcal{S}}$ obtained via the Set Cover construction is
consistent. This fact guarantees the existence of a correct ensemble and is the starting
point for the chain of lemmas leading to the NP-hardness result.

## Statement

For any Set Cover instance $(W, \mathcal{S})$ with $\bigcup \mathcal{S} = W$, the decision
table $\mathbb{A}_{\mathcal{S}}$ constructed according to [the construction](../cdbe-kp-np-complete/set-cover-construction.md)
is consistent.

## Proof

Each object in $U_{\mathcal{S}}$ has decision value $0$ and at least one $1$ on the
conditional attributes from $A_{\mathcal{S}}$ -- this follows from the fact that
$\bigcup \mathcal{S} = W$. On the other hand, $u_*$ is the only object with decision value
$1$, and it has $0$ on every conditional attribute. Hence any two objects that are
$A_{\mathcal{S}}$-indiscernible must share the same decision value (the equivalent
[formulation](../../concepts/consistency.md#equivalent-formulations) of consistency), so
$\mathbb{A}_{\mathcal{S}}$ is consistent.

## Remarks

Consistency of $\mathbb{A}_{\mathcal{S}}$ combined with the earlier result that a correct
ensemble exists iff the table is consistent ([link](../../correct-ensemble-iff-dectab-consistent.md))
implies that at least one correct ensemble of decision bireducts exists for
$\mathbb{A}_{\mathcal{S}}$. The subsequent lemmas investigate the structure of such
ensembles.
