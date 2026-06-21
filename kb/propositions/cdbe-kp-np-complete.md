---
id: prop-cdbe-kp-np-complete
type: proposition
status: draft
created: 2026-06-20
updated: 2026-06-20
tags: [complexity, bireducts, ensemble]
requires:
  - concept-bireduct-ensemble
  - concept-decision-table
  - concept-consistency
  - prop-correct-ensemble-iff-dectab-consistent
  - prop-set-cover-problem
  - prop-set-cover-construction
see_also:
  - prop-ensemble-np-hard
source: src-reduct-matrix-ensembles
---

# NP-Completeness of CDBEkP

The Correct Decision Bireduct Ensemble of Size $k$ Problem (CDBEkP) is NP-complete.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be a decision table and $k \geq 0$ be an integer. The problem
CDBEkP -- deciding whether there exists a correct ensemble of decision bireducts for $\mathbb{A}$
with description length at most $k$ -- is NP-complete.

## Proof Strategy

The proof reduces from the **Set Cover** problem. Given a Set Cover instance $(W, \mathcal{S})$, we
construct a consistent decision table $\mathbb{A}_{\mathcal{S}}$ and a bound $k'$ such that a set
cover of size at most $s$ exists for $(W, \mathcal{S})$ if and only if a correct ensemble with
description length at most $k'$ exists for $\mathbb{A}_{\mathcal{S}}$.

The proof proceeds through a series of intermediate lemmas establishing structural properties of
bireducts in the transformed table $\mathbb{A}_{\mathcal{S}}$. Auxiliary lemmas are collected in
[cdbe-kp-np-complete/](cdbe-kp-np-complete/).

## Proof

See the auxiliary lemmas in [cdbe-kp-np-complete/](cdbe-kp-np-complete/) for the detailed
construction and verification.

## Remarks

The source (`tmp/pub/main.tex`) labels this as NP-complete with a TODO noting that NP membership
depends on whether a polynomial-time verification procedure can be established. The reduction from
Set Cover establishing NP-hardness is in Section 4; Section 5 (optimization problem, SCDBEP NP-hard)
is empty in the source.

Local notation for the Set Cover reduction (symbols specific to this proof) is collected in
[cdbe-kp-np-complete/notation.md](cdbe-kp-np-complete/notation.md) rather than in the global
`kb/notation.md`.
