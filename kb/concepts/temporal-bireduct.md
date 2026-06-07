---
id: concept-temporal-bireduct
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, bireducts]
requires: [concept-decision-table, concept-decision-bireduct]
see_also: [concept-decision-bireduct]
source: src-thesis-phd
---

# Temporal Bireduct

A temporal decision bireduct is a decision bireduct whose covered object set forms a continuous range
with respect to a natural ordering of the universe -- useful in data stream scenarios where the full
dataset is not available upfront.

## Definition

Let $\mathbb{A} = (U, A \cup \{d\})$ be given where $U$ is naturally ordered with elements indexed by
integers. A pair $(X, B)$ where $X = \{u_i \in U : i \in \{first, \ldots, last\}\}$ is a continuous
range of objects is a **temporal decision bireduct** if and only if:

1. $B \Rrightarrow_X d$ (the inexact functional dependency holds).
2. No proper subset $C \subsetneq B$ satisfies $C \Rrightarrow_X d$.
3. $B \Rrightarrow_{X'} d$ does **not** hold for $X'$ extended backward
   ($\{u_{first-1}, \ldots, u_{last}\}$) nor forward ($\{u_{first}, \ldots, u_{last+1}\}$).

## Motivation

Temporal bireducts address scenarios where data arrives as a stream and $U$ cannot be fully accessed
at any moment. Instead of fixing a window size arbitrarily, temporal bireducts adaptively adjust data
intervals based on currently observed attribute dependencies.

## Computation Sketch

A heuristic extraction procedure processes objects sequentially:

1. Maintain a buffer $X$ (objects processed so far) and attribute subset $B$.
2. If the next object $u_i$ can be added while preserving $B \Rrightarrow_{X \cup \{u_i\}} d$, add
   it.
3. Otherwise, save $(X, B)$ as a temporal bireduct, reset $B$ to a chosen $A' \subseteq A$, remove
   oldest objects from $X$ until the dependency holds again, and heuristically reduce $B$.

All saved pairs are temporal bireducts.

## Remarks

The approach resembles micro-clustering for data streams and can produce bireducts with no "gaps" in
the object sequence. By working with multiple diversified subsets $A' \subseteq A$, one can observe
the most representative changes of temporal bireducts over time. Frequently occurring attribute
subsets across temporal bireducts may indicate robust decision models.
