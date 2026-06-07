---
id: concept-indiscernibility
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, indiscernibility]
requires: [concept-decision-table]
see_also:
  [concept-discernibility, concept-approximations, concept-consistency, concept-decision-reduct]
source: src-thesis-phd
---

# Indiscernibility

The indiscernibility relation is the fundamental equivalence relation of rough set theory. It groups
objects that cannot be distinguished from each other based on a given subset of attributes.

## Indiscernibility Relation

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. An attribute subset $B \subseteq A \cup \{d\}$
determines a binary relation $IND(B)$ on $U$:

$$
u \; IND(B) \; u' \quad \Longleftrightarrow \quad \forall_{a \in B}\; a(u) = a(u')
$$

We say that $u$ and $u'$ are indiscernible by attributes of $B$.

### Proposition: Equivalence Relation

$IND(B)$ is an equivalence relation. See the [full
proof](../propositions/indiscernibility-equivalence-relation.md).

### Quotient Set and Equivalence Classes

As an equivalence relation, $IND(B)$ partitions $U$ into a quotient set $U/B$ (also denoted
$U/IND(B)$). An element of $U/B$ containing object $u \in U$ is denoted $[u]_B$ (or
$[u]_{IND(B)}$).

Decision classes are the elements of $U/\{d\}$, denoted $X^{\langle 1 \rangle}, \ldots,
X^{\langle \lvert V_d \rvert \rangle}$. The notation $X^{\langle d = v \rangle}$ denotes the class
of objects with decision value $v \in V_d$.

## Discernibility Relation

The discernibility relation $DIS(B)$ is the complement of $IND(B)$:

$$
u \; DIS(B) \; u' \quad \Longleftrightarrow \quad \neg(u \; IND(B) \; u')
\quad \Longleftrightarrow \quad \exists_{a \in B}\; a(u) \neq a(u')
$$

We say that $B$ discerns $u$ and $u'$ (or that $u, u'$ are discerned by $B$).

## Remarks

Indiscernibility is the fundamental building block of rough set theory. It enables the definition of
[lower and upper approximations](../concepts/approximations.md), the [positive
region](../concepts/positive-region.md), and [decision reducts](../concepts/decision-reduct.md).

The equivalence class notation $[u]_B$ and quotient set $U/B$ are used throughout the theory.
The generic variable $E \in U/B$ is often used to denote a single equivalence class.
