---
id: prop-decision-bireduct-boolean-formula
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [core, bireducts, boolean-reasoning]
requires:
  [concept-decision-bireduct,
   concept-formulae]
see_also:
  [prop-decision-reduct-boolean-formula,
   prop-gamma-decision-bireduct-boolean-formula,
   prop-decision-table-diagonal]
source: tmp/phd/thesis.tex
---

# Decision Bireduct Boolean Formula Characterisation

Decision bireducts correspond to prime implicants of a Boolean formula $\tau_{bi}$ that extends the
classical discernibility formula with propositional variables for objects.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. Consider the following Boolean formula with
propositional variables $\overline{i}$ (for $i = 1, \ldots, \lvert U \rvert$) and $\overline{a}$
(for $a \in A$):

$$
\tau_{bi} = \bigwedge_{\substack{u_i, u_j \in U \\ i < j,\; d(u_i) \neq d(u_j)}}
  \left(
    \overline{i} \lor \overline{j} \lor
    \bigvee_{\substack{a \in A \\ a(u_i) \neq a(u_j)}} \overline{a}
  \right)
$$

A pair $(X, B)$, where $X \subseteq U$ and $B \subseteq A$, is a decision bireduct if and only if
the Boolean formula

$$
P = \bigwedge_{u_i \in U \setminus X} \overline{i} \land \bigwedge_{a \in B} \overline{a}
$$

is a prime implicant for $\tau_{bi}$.

## Proof

A product term $P$ is an implicant of a Boolean formula $\tau$ if $P$ being true always implies
$\tau$ is true. A prime implicant is an implicant minimal with regard to inclusion -- removing any
literal causes it to no longer be an implicant.

Let $B \subseteq A$ and $X \subseteq U$ be given. Consider
$P = \bigwedge_{u_i \in U \setminus X} \overline{i} \land \bigwedge_{a \in B} \overline{a}$.

### Step 1: Functional Dependency Equivalence

First, we show that:

$$
B \Rrightarrow_X d \;\Longleftrightarrow\; P \text{ is an implicant for } \tau_{bi}
$$

**($\Rightarrow$)** Suppose that $P$ is not an implicant for $\tau_{bi}$. Hence, there exists a
valuation of propositional variables for which $P$ is true but $\tau_{bi}$ is false. Thus, there is
at least one clause of the form

$$
f_k = \left(\overline{i_k} \lor \overline{j_k} \lor \bigvee_{\substack{a \in A \\ a(u_{i_k}) \neq a(u_{j_k})}} \overline{a}\right)
$$

where $u_{i_k}, u_{j_k} \in U$ and $d(u_{i_k}) \neq d(u_{j_k})$, which is false for the considered
valuation. As $f_k$ is a disjunction, all its elements must be assigned false. Since $P$ is true
and both $\overline{i_k}$ and $\overline{j_k}$ are false, neither $\overline{i_k}$ nor
$\overline{j_k}$ is part of $P$. Since $P$ contains variables corresponding to all objects in
$U \setminus X$, we know that $u_{i_k}, u_{j_k} \in X$. We also know that $P$ cannot contain
variables corresponding to attributes for which $u_{i_k}$ and $u_{j_k}$ have different values, i.e.,
for all $a \in A$ such that $a(u_{i_k}) \neq a(u_{j_k})$, we know that $a \notin B$. This means
that there are at least two objects $u_{i_k}, u_{j_k} \in X$ such that $d(u_{i_k}) \neq d(u_{j_k})$
which are not discerned by $B$. Therefore, $B \Rrightarrow_X d$ does not hold.

**($\Leftarrow$)** Suppose that $B \Rrightarrow_X d$ does not hold. This means that there exists at
least one pair of objects $u_{i_k}, u_{j_k} \in X$ such that $d(u_{i_k}) \neq d(u_{j_k})$ which is
not discerned by $B$. Consider the corresponding clause $f_k$. Assign false to the variables in
$f_k$ and true to all others. For such valuation, $P$ is true because it does not share any elements
with $f_k$ (the objects are in $X$, so their variables are not in $P$; the attributes do not discern
the pair, so their variables are not in $P$). On the other hand, $\tau_{bi}$ is false. Thus, $P$ is
not an implicant for $\tau_{bi}$.

### Step 2: Bireduct Equivalence

Using Step 1, we show the desired equivalence:

$$
(X, B) \text{ is a decision bireduct } \;\Longleftrightarrow\; P \text{ is a prime implicant for } \tau_{bi}
$$

**($\Rightarrow$)** Suppose that $P$ is not a prime implicant for $\tau_{bi}$. We have two
possibilities:

1. $P$ is not an implicant -- this is resolved by Step 1 (it would mean $B \Rrightarrow_X d$ does
   not hold, so $(X, B)$ is not a bireduct).

2. $P$ is an implicant but not a prime implicant. This means we can remove at least one element of
   $P$ while preserving the implicant property. There are two cases:

   - A variable that can be removed corresponds to an object $u_{i_k} \in U \setminus X$. Then
     $P' = \bigwedge_{u_i \in U \setminus (X \cup \{u_{i_k}\})} \overline{i} \land \bigwedge_{a \in B} \overline{a}$
     is an implicant for $\tau_{bi}$. From Step 1, $B \Rrightarrow_{X \cup \{u_{i_k}\}} d$, so
     $(X, B)$ is not a decision bireduct (object maximality violated).

   - A variable that can be removed corresponds to an attribute $a_k \in B$. Then
     $P' = \bigwedge_{u_i \in U \setminus X} \overline{i} \land \bigwedge_{a \in B \setminus \{a_k\}} \overline{a}$
     is an implicant for $\tau_{bi}$. From Step 1, $(B \setminus \{a_k\}) \Rrightarrow_X d$, so
     $(X, B)$ is not a decision bireduct (attribute irreducibility violated).

**($\Leftarrow$)** Suppose that $(X, B)$ is not a decision bireduct for $\mathbb{A}$. We have three
possibilities:

1. $B \not\Rrightarrow_X d$. Then from Step 1, $P$ is not an implicant.

2. There exists a proper subset $B' \subsetneq B$ such that $B' \Rrightarrow_X d$. Then from Step 1,
   $P' = \bigwedge_{u_i \in U \setminus X} \overline{i} \land \bigwedge_{a \in B'} \overline{a}$ is
   an implicant for $\tau_{bi}$. Therefore, $P$ is not a prime implicant.

3. There exists a proper superset $X' \supsetneq X$ such that $B \Rrightarrow_{X'} d$. Then from
   Step 1, $P' = \bigwedge_{u_i \in U \setminus X'} \overline{i} \land \bigwedge_{a \in B} \overline{a}$
   is an implicant for $\tau_{bi}$. Therefore, $P$ is not a prime implicant.

All cases together finish the proof.

## Remarks

This result shows that attributes and objects are to some extent equally important when constructing
decision bireducts. The number of decision bireducts is usually far higher than the number of
standard decision reducts. For the golf dataset, there are only 2 decision reducts but many more
decision bireducts (see [Golf All Bireducts](../examples/golf-all-bireducts.md)).

The Boolean formula $\tau_{bi}$ can be transformed from CNF to DNF to enumerate all decision
bireducts as prime implicants. This provides a direct method for bireduct computation using Boolean
reasoning techniques.
