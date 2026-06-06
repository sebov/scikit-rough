---
id: prop-temporal-bireduct-computation
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [bireducts, algorithms, temporal, data-streams]
requires:
  [concept-decision-table,
   concept-decision-bireduct,
   concept-temporal-bireduct]
see_also:
  [concept-temporal-bireduct,
   concept-decision-bireduct,
   prop-decision-bireduct-ordering]
source: tmp/phd/thesis.tex
---

# Temporal Bireduct Computation via Streaming Buffer

A sequential processing algorithm with a sliding buffer of objects produces temporal bireducts for
ordered data streams. Every temporal bireduct is achievable by some choice of attribute subset $A'$.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be given with $U$ naturally ordered by integer indices. Select
an arbitrary subset $A' \subseteq A$, set $X = \emptyset$ and $B = \emptyset$. For each consecutive
$i$-th object in $U$:

1. If $B \Rrightarrow_{X \cup \{u_i\}} d$, then add $u_i$ to $X$.
2. Otherwise, save $(X, B)$, add $u_i$ to $X$, and:
   - Set $B = A'$ and remove the oldest objects from $X$ until $B \Rrightarrow_X d$.
   - Heuristically reduce redundant attributes under the constraint $B \Rrightarrow_X d$.

Then all pairs $(X, B)$ saved during the procedure are temporal bireducts for $\mathbb{A}$.
Moreover, each temporal bireduct can be obtained as one of the saved pairs for some $A' \subseteq A$,
regardless of the heuristic reduction method used.

## Background

A temporal bireduct $(X, B)$ differs from an ordinary decision bireduct in the maximality condition:
instead of requiring $X$ to be maximal in all directions, it requires $X$ to be a continuous range
$\{u_{first}, \ldots, u_{last}\}$ that cannot be extended backward (to $u_{first-1}$) or forward (to
$u_{last+1}$) while preserving the functional dependency.

The streaming algorithm processes objects in arrival order, maintaining a buffer $X$ of the most
recent objects and an attribute subset $B$. When a new object causes the dependency to break, the
algorithm saves the current buffer as a temporal bireduct and restarts with a fresh $B = A'$.

## Proof

The proof has two parts.

### Part 1: All Saved Pairs Are Temporal Bireducts

Consider a pair $(X, B)$ where $X = \{u_{first}, \ldots, u_{last}\}$ that was saved in step 2 of
the algorithm.

**Functional dependency.** By construction, $B \Rrightarrow_X d$ holds at the moment of saving
(the algorithm resets $B$ and trims $X$ until this condition is true).

**Forward non-extendability.** The pair is saved precisely when $u_{last+1}$ arrives and cannot be
added while preserving the dependency. Therefore $B \not\Rrightarrow_{\{u_{first}, \ldots,
u_{last+1}\}} d$.

**Backward non-extendability.** The oldest objects are removed from the buffer only when a newly
arrived object cannot be handled together with some current elements of $X$ even when using the
full $A'$. Thus, at the moment of saving, adding $u_{first-1}$ would also break the dependency.
Formally: $B \not\Rrightarrow_{\{u_{first-1}, \ldots, u_{last}\}} d$.

> **Proof gap (flagged).** This argument has a subtle incompleteness. The algorithm removes ALL
> oldest objects until $A'$ determines $d$, and $u_{first}$ survives -- so $u_{first-1}$ was
> indeed essential to at least one conflict under $A'$. However, the object that $u_{first-1}$
> conflicted with may itself have been removed in a *subsequent* reset before save time. If that
> conflicting partner is no longer in the buffer, adding $u_{first-1}$ back might not break the
> dependency, making the saved pair fail backward non-extendability.
>
> A complete proof would need to show that either the conflicting partner of $u_{first-1}$
> persists in the buffer until save time, or that removal of the partner triggers a cascade that
> also re-exposes the original conflict. The thesis's argument sketches the right intuition
> (objects causing trouble get removed, and the trouble-maker of the trouble-maker is still
> there) but does not close this chain formally.

**Attribute irreducibility.** The heuristic reduction in step 2b removes redundant attributes
while preserving the dependency. Therefore no proper subset $C \subsetneq B$ satisfies $C
\Rrightarrow_X d$.

All four conditions of a temporal bireduct are satisfied.

### Part 2: Every Temporal Bireduct Is Achievable

Let $(X, B)$ with $X = \{u_{first}, \ldots, u_{last}\}$ be an arbitrary temporal bireduct. Set
$A' = B$.

Consider the state of the algorithm when processing $u_{first}$. The buffer at that point contains
some objects $\{u_{older}, \ldots, u_{first}\}$ with $older \le first$. Each subsequent object up
to $u_{last}$ will be added without needing to remove $u_{first}$, because if removal were
necessary, then $B$ could not determine $d$ on $\{u_{first}, \ldots, u_{last}\}$ -- contradicting
the temporal bireduct property.

When $u_{last}$ is added, all objects older than $u_{first}$ that remain in the buffer will be
removed. If any such object persisted, then $B$ would determine $d$ on $\{u_{first-1}, \ldots,
u_{last}\}$, contradicting backward non-extendability of the temporal bireduct.

Finally, on processing $u_{last+1}$, the algorithm will need to remove $u_{first}$ from the
buffer. If it did not (i.e., $B \Rrightarrow_{\{u_{first}, \ldots, u_{last+1}\}} d$ held), this
would contradict forward non-extendability. Removal of $u_{first}$ triggers the save in step 2,
producing exactly $(X, B)$.

## Remarks

The algorithm is heuristic in two places: the choice of $A' \subseteq A$ and the attribute
reduction method in step 2b. These choices influence which temporal bireducts are produced and
their size characteristics (attribute-to-object ratio).

Working with multiple diversified subsets $A'$ enables observing representative changes of
temporal bireducts over time. Frequently recurring attribute subsets across temporal bireducts may
indicate robust decision models resistant to concept drift.
