---
id: concept-np-hardness-foundations
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, complexity]
requires: []
see_also:
  [concept-approximate-decision-reduct,
   concept-epsilon-decision-bireduct,
   concept-bireduct-ensemble]
source: tmp/phd/thesis.tex
---

# NP-Hardness Foundations

Several optimization problems in rough set theory -- including finding minimal approximate reducts,
minimal $\varepsilon$-bireducts, and simplest correct ensembles -- are proved NP-hard via polynomial
reduction from graph-theoretic problems. This file collects the graph definitions used across those
proofs.

## Graph

A graph is an ordered pair $\mathbb{G} = (\mathbb{V}, \mathbb{E})$, where $\mathbb{V}$ is the set of
vertices (nodes) and $\mathbb{E} \subseteq \mathbb{V} \times \mathbb{V}$ is the set of edges. A graph
is **non-directed** if the relation $\mathbb{E}$ is symmetric.

## Dominating Set

Let a non-directed graph $\mathbb{G} = (\mathbb{V}, \mathbb{E})$ be given. A subset
$\mathbb{W} \subseteq \mathbb{V}$ is a dominating set for $\mathbb{G}$ if and only if:

$$
Cov_{\mathbb{G}}(\mathbb{W}) = \mathbb{V}
$$

where

$$
Cov_{\mathbb{G}}(\mathbb{W}) = \mathbb{W} \cup \{v \in \mathbb{V} :
  \exists_{w \in \mathbb{W}}\; (v, w) \in \mathbb{E}\}
$$

is the set of vertices that either belong to $\mathbb{W}$ or are adjacent to at least one member of
$\mathbb{W}$.

### Minimal Dominating Set Problem

The Minimal Dominating Set Problem is the optimization problem of finding a dominating set with
minimum cardinality for a given undirected graph. This problem is NP-hard (Garey & Johnson, 1979).

## $\alpha$-Dominating Set

Let $\alpha \in (0, 1]$ and a non-directed graph $\mathbb{G} = (\mathbb{V}, \mathbb{E})$ be given.
A subset $\mathbb{W} \subseteq \mathbb{V}$ is an $\alpha$-dominating set for $\mathbb{G}$ if:

$$
\frac{\lvert Cov_{\mathbb{G}}(\mathbb{W}) \rvert}{\lvert \mathbb{V} \rvert} \geq \alpha
$$

The Minimal $\alpha$-Dominating Set Problem (finding a minimal such set) is NP-hard for any
$\alpha \in (0, 1]$ (Slezak, 2000).

## Remarks

These graph problems serve as the base for polynomial reductions in NP-hardness proofs throughout
the dissertation:

- Minimal $\gamma$-decision $\varepsilon$-reduct is NP-hard via reduction from Minimal Dominating
  Set.
- Minimal $M$-decision and $R$-decision $\varepsilon$-reducts are NP-hard via reduction from Minimal
  $\alpha$-Dominating Set.
- Minimal $\varepsilon$-decision bireduct is NP-hard via reduction from Minimal $M$-decision
  $\varepsilon$-reduct.
- Simplest correct decision bireduct ensemble is NP-hard via reduction from Minimal Dominating Set.
