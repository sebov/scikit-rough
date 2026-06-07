---
id: prop-decision-table-diagonal
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [core, bireducts, algorithms]
requires:
  [concept-decision-bireduct,
   concept-decision-reduct,
   concept-decision-table]
see_also:
  [prop-decision-bireduct-boolean-formula,
   concept-decision-bireduct,
   prop-decision-reduct-iff-bireduct]
source: src-thesis-phd
---

# Diagonal Table Transformation for Bireducts

Decision bireducts can be computed by transforming the problem to standard decision reducts on a
modified table with diagonal attributes. Each object gets a unique identifier attribute.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. Consider a new decision table
$\mathbb{A}^{\boxbslash} = (U, (A \cup A^{\boxbslash}) \cup \{d\})$, where the number of objects in
$U$ and their values for original attributes in $A$ remain unchanged, and where the new attributes
$A^{\boxbslash} = \{a^{\boxbslash}_1, \ldots, a^{\boxbslash}_{\lvert U \rvert}\}$ are defined as
follows:

$$
a^{\boxbslash}_j(u_i) =
\begin{cases}
  1 & \text{if } i = j \\
  0 & \text{otherwise}
\end{cases}
$$

Then a pair $(X, B)$, where $B \subseteq A$ and $X \subseteq U$, is a decision bireduct for
$\mathbb{A}$ if and only if $B \cup A^{\boxbslash}_X$, for

$$
A^{\boxbslash}_X = \{a^{\boxbslash}_i \in A^{\boxbslash} : u_i \notin X\}
$$

is a decision reduct for $\mathbb{A}^{\boxbslash}$.

## Proof

**($\Rightarrow$)** Let a decision bireduct $(X, B)$ for $\mathbb{A}$ be given. We need to show that
for $B \cup A^{\boxbslash}_X$, the decision reduct conditions for $\mathbb{A}^{\boxbslash}$ are met.

First, we know that $B$ discerns all objects from $X$ with different decision values and -- because
it is non-extendable -- no other object from $U \setminus X$ can be added. Therefore, to ensure
discernibility among all objects with different decision values in $\mathbb{A}^{\boxbslash}$, we
need to add some attributes from $A^{\boxbslash}$. A given $a^{\boxbslash}_i \in A^{\boxbslash}$
discerns object $u_i$ from all other objects. Thus, it is enough to take $B \cup A^{\boxbslash}_X$
to ensure discernibility in $\mathbb{A}^{\boxbslash}$: objects in $X$ are discerned by $B$, and
objects outside $X$ are discerned by their diagonal attributes.

Secondly, we have to show that $B \cup A^{\boxbslash}_X$ is irreducible:

- We cannot remove any attribute from $A^{\boxbslash}_X$ because -- otherwise -- some object
  $u_i \notin X$ would become indiscernible from objects in $X$ that share its decision class,
  violating discernibility in $\mathbb{A}^{\boxbslash}$.

- We cannot remove any attribute from $B$. If we could, i.e., if there is $B' \subsetneq B$ such
  that $B' \cup A^{\boxbslash}_X$ discerns all objects with different decision values in
  $\mathbb{A}^{\boxbslash}$, then we would have $B' \Rrightarrow_X d$ and $(X, B)$ would not be a
  decision bireduct.

**($\Leftarrow$)** Let a decision reduct $B \cup A^{\boxbslash}_X$ for $\mathbb{A}^{\boxbslash}$ be
given. We need to show that $(X, B)$ is a decision bireduct in $\mathbb{A}$.

Clearly, $B \Rrightarrow_X d$ (objects in $X$ with different decisions are discerned by $B$, since
they are discerned in $\mathbb{A}^{\boxbslash}$ and diagonal attributes don't help discern objects
within $X$).

Suppose that $B$ can be reduced, i.e., there is $B' \subsetneq B$ such that $B' \Rrightarrow_X d$.
However, in such a case $B' \cup A^{\boxbslash}_X$ would discern all objects with different decision
values in $\mathbb{A}^{\boxbslash}$, so $B \cup A^{\boxbslash}_X$ would not be a decision reduct --
a contradiction.

Secondly, suppose that we can extend $X$ by some object $u_i \in U \setminus X$, i.e., there is
$B \Rrightarrow_{X \cup \{u_i\}} d$. However, it would imply that $B \cup A^{\boxbslash}_X$ is not
a decision reduct because we would be able to reduce it by removing the attribute
$a^{\boxbslash}_i$ -- a contradiction.

## Remarks

This transformation allows applying standard reduct algorithms to the bireduct search problem.
However, it should be treated as a starting point for developing more efficient algorithms, because
decision tables of the form $\mathbb{A}^{\boxbslash}$ cannot be explicitly materialized for large
data (the number of attributes grows linearly with $\lvert U \rvert$).

On the other hand, this representation may be an inspiration for adapting attribute clustering
techniques developed for high-dimensional data to the problem of searching for ensembles of decision
bireducts.

For an illustrative example of the diagonal transformation, see
[Golf All Bireducts](../examples/golf-all-bireducts.md).
