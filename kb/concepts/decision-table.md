---
id: concept-decision-table
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, decision-table]
requires: []
see_also:
  [concept-indiscernibility, concept-formulae, concept-decision-rule, concept-decision-reduct]
source: tmp/phd/thesis.tex
---

# Decision Table

A decision table $\mathbb{A} = (U, A \cup \{d\})$ is the standard tabular data representation used
throughout rough set theory, pairing a universe of objects with conditional and decision attributes.

## Definition

Let $\mathbb{A} = (U, A \cup \{d\})$ be a pair of non-empty sets, where:

- $U$ is a universe of objects.
- $A \cup \{d\}$ is a set consisting of attributes such that every $a \in A \cup \{d\}$ is a
  function $a : U \to V_a$, where $V_a$ denotes $a$'s codomain and is called the value set of $a$.
- The distinguished attribute $d$, such that $d \notin A$, is called a decision attribute.
- The elements of $A$ are called conditional attributes.

In practice, both $U$ and $A \cup \{d\}$ are finite. The values $v_d \in V_d$ of the decision
attribute correspond to decision classes that we aim to describe using the conditional attributes in
$A$.

For notational convenience, elements of $U$ are often referred to by their ordinal numbers
$i = 1, \ldots, \lvert U \rvert$, with the shorthand $\{u_1, u_4, u_5\}$ meaning the subset of
objects $u_1, u_4, u_5$ in $U$.

Decision classes are denoted by $X^{\langle k \rangle}$ for $k = 1, \ldots, \lvert V_d \rvert$, where
$X^{\langle d = v \rangle}$ denotes the class of objects with decision value $v \in V_d$.

## Intuition

A decision table is the fundamental data structure of rough set theory -- it is essentially a table
where rows are objects, columns are attributes (features), and one special column is the decision
(label). The conditional attributes describe the objects; the decision attribute is what we want to
predict or explain.

## Example

The well-known golf dataset shown below is a decision table with $\lvert U \rvert = 14$,
$A = \{\text{Outlook}, \text{Temperature}, \text{Humidity}, \text{Wind}\}$, and
$d = \text{Play}$ (with $V_d = \{\text{yes}, \text{no}\}$).

| ID | Outlook  | Temperature | Humidity | Wind   | Play |
| -- | -------- | ----------- | -------- | ------ | ---- |
| 1  | sunny    | hot         | high     | weak   | no   |
| 2  | sunny    | hot         | high     | strong | no   |
| 3  | overcast | hot         | high     | weak   | yes  |
| 4  | rain     | mild        | high     | weak   | yes  |
| 5  | rain     | cool        | normal   | weak   | yes  |
| 6  | rain     | cool        | normal   | strong | no   |
| 7  | overcast | cool        | normal   | strong | yes  |
| 8  | sunny    | mild        | high     | weak   | no   |
| 9  | sunny    | cool        | normal   | weak   | yes  |
| 10 | rain     | mild        | normal   | weak   | yes  |
| 11 | sunny    | mild        | normal   | strong | yes  |
| 12 | overcast | mild        | high     | strong | yes  |
| 13 | overcast | hot         | normal   | weak   | yes  |
| 14 | rain     | mild        | high     | strong | no   |

## Remarks

The decision table is the starting point for all further rough set concepts. The
[indiscernibility](../concepts/indiscernibility.md) relation groups objects that look the same under
a given attribute subset. [Decision rules](../concepts/decision-rule.md) can be extracted to
classify new objects. [Decision reducts](../concepts/reducts.md) identify minimal subsets of
attributes that preserve the decision information.

The notion originates from Pawlak's seminal works on rough sets (Pawlak, 1991; Pawlak, 2007).
