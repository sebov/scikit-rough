---
id: ex-nphard-construction-tables
type: example
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [example, np-hardness, graph, construction]
requires:
  [concept-np-hardness-foundations,
   prop-relative-gamma-epsilon-reduct-np-hard,
   prop-relative-m-epsilon-reduct-np-hard]
see_also:
  [prop-relative-gamma-epsilon-reduct-np-hard,
   prop-relative-m-epsilon-reduct-np-hard,
   concept-np-hardness-foundations]
source: src-thesis-phd
---

# NP-Hardness Construction -- Example Decision Tables

The same 8-vertex graph $\mathbb{G}$ used to illustrate both the $\gamma$ and $M$ reduction
constructions from the NP-hardness proofs.

## The Graph

An 8-vertex graph with vertices $1, \ldots, 8$ arranged in two "rings": vertices $1,2,3,4$ form a
cycle, vertices $5,6,7,8$ form another cycle, and connecting edges are $(1,5)$, $(2,6)$, $(3,7)$,
$(4,8)$.

## Gamma Construction

Table $\mathbb{A}^{\varepsilon}_{\mathbb{G}}$ for the reduction in
[NP-Hardness of Minimal Relative Gamma-Decision Epsilon-Reduct](../propositions/relative-gamma-epsilon-reduct-np-hard.md).
Binary attributes $a_1, \ldots, a_8$ (one per vertex). Vertex-objects $u_1, \ldots, u_8$ have
$a_j(u_i) = 1$ iff $i=j$ or $(v_i, v_j) \in \mathbb{E}$. Auxiliary objects $u_{9}, \ldots,
u_{8+t(\varepsilon)}$ have all zeros and decision 1.

| $u_i$ | $a_1$ | $a_2$ | $a_3$ | $a_4$ | $a_5$ | $a_6$ | $a_7$ | $a_8$ | $d$ |
|:------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---:|
| $u_1$ | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| $u_2$ | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| $u_3$ | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| $u_4$ | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| $u_5$ | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
| $u_6$ | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| $u_7$ | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 0 |
| $u_8$ | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| $u_9$ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| $u_{9+1}$ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ | $\cdots$ |
| $u_{9+t(\varepsilon)}$ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

## M Construction

Table $\mathbb{A}^{\varepsilon*}_{\mathbb{G}}$ for the reduction in
[NP-Hardness of Minimal Relative M-Decision Epsilon-Reduct](../propositions/relative-m-epsilon-reduct-np-hard.md),
with $\varepsilon = 0.6$. Then $m(\varepsilon) = \lfloor (1-0.6)^{-1} + 1 \rfloor = \lfloor 2.5 +
1 \rfloor = 3$, so each of the 8 vertices produces $3$ objects, giving $|U| = 24$. Decision values
cycle $1,2,3,1,2,3,\ldots$ in chunks of 3.

| $u_i$ | $a_1$ | $a_2$ | $a_3$ | $a_4$ | $a_5$ | $a_6$ | $a_7$ | $a_8$ | $d$ |
|:------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---:|
| $u_1$ | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| $u_2$ | 2 | 2 | 1 | 2 | 2 | 1 | 1 | 1 | 2 |
| $u_3$ | 3 | 3 | 1 | 3 | 3 | 1 | 1 | 1 | 3 |
| $u_4$ | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 1 |
| $u_5$ | 5 | 5 | 5 | 4 | 4 | 5 | 4 | 4 | 2 |
| $u_6$ | 6 | 6 | 6 | 4 | 4 | 6 | 4 | 4 | 3 |
| $u_7$ | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 1 |
| $u_8$ | 7 | 8 | 8 | 8 | 7 | 7 | 8 | 7 | 2 |
| $u_9$ | 7 | 9 | 9 | 9 | 7 | 7 | 9 | 7 | 3 |
| $u_{10}$ | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 1 |
| $u_{11}$ | 11 | 10 | 11 | 11 | 10 | 10 | 10 | 11 | 2 |
| $u_{12}$ | 12 | 10 | 12 | 12 | 10 | 10 | 10 | 12 | 3 |
| $u_{13}$ | 13 | 13 | 13 | 13 | 13 | 13 | 13 | 13 | 1 |
| $u_{14}$ | 14 | 13 | 13 | 13 | 14 | 14 | 13 | 14 | 2 |
| $u_{15}$ | 15 | 13 | 13 | 13 | 15 | 15 | 13 | 15 | 3 |
| $u_{16}$ | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 1 |
| $u_{17}$ | 16 | 17 | 16 | 16 | 17 | 17 | 17 | 16 | 2 |
| $u_{18}$ | 16 | 18 | 16 | 16 | 18 | 18 | 18 | 16 | 3 |
| $u_{19}$ | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 1 |
| $u_{20}$ | 19 | 19 | 20 | 19 | 19 | 20 | 20 | 20 | 2 |
| $u_{21}$ | 19 | 19 | 21 | 19 | 19 | 21 | 21 | 21 | 3 |
| $u_{22}$ | 22 | 22 | 22 | 22 | 22 | 22 | 22 | 22 | 1 |
| $u_{23}$ | 22 | 22 | 22 | 23 | 23 | 22 | 23 | 23 | 2 |
| $u_{24}$ | 22 | 22 | 22 | 24 | 24 | 22 | 24 | 24 | 3 |

## Remarks

In the gamma construction, the number of auxiliary objects $t(\varepsilon)$ depends on
$\varepsilon$ and $|\mathbb{V}|$. For $\varepsilon=0.6$ and $|\mathbb{V}|=8$:
$t = \lfloor \frac{8 \cdot 0.6}{0.4} + 1 \rfloor = \lfloor 12 + 1 \rfloor = 13$.

In the M construction, $m(\varepsilon) = 3$ for $\varepsilon=0.6$. Objects are grouped in chunks
of $m(\varepsilon)$ (3 objects per vertex). Within each chunk, the decision cycles $1,2,3$. The
attribute values encode adjacency: $a_j(u_i) = i$ when $\operatorname{chunk}(i)=j$ or
$(v_{\operatorname{chunk}(i)}, v_j) \in \mathbb{E}$, and the default value
$(\operatorname{chunk}(i)-1)m(\varepsilon)+1$ otherwise.
