---
id: prop-bireduct-objects-and-rules
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [core, bireducts, rules]
requires:
  [concept-decision-bireduct,
   concept-decision-rule,
   concept-indiscernibility]
see_also:
  [concept-decision-rule,
   ex-golf-bireduct-rules,
   concept-gamma-decision-bireduct]
source: src-thesis-phd
---

# Bireduct Objects and Rules

Each decision bireduct $(X, B)$ induces a well-defined collection of deterministic decision rules
whose supports sum to exactly $X$. This provides a rule-based interpretation of bireducts.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be given and $(X, B)$ be a decision bireduct for
$\mathbb{A}$. The following statements are true:

1. For each $E \in U/B$, all objects in $X \cap E$ have the same decision value, further denoted
   as $v_{d(X \cap E)} \in V_d$.
2. For each $E \in U/B$, all objects in $E$ which have value $v_{d(X \cap E)} \in V_d$ on $d$
   are contained in $X$.
3. $X$ equals the union of supports of the following set of decision rules:

$$
Rules(X, B) =
\left\{
  \bigwedge_{a \in B} (a = a(u)) \Rightarrow (d = d(u))
  : u \in X
\right\}
$$

## Proof

**(1.)** Suppose that $X \cap E$ contains objects from more than one decision class, i.e., there
are objects $u_i, u_j \in X \cap E$ such that $d(u_i) \neq d(u_j)$. It would violate the
definition of decision bireduct -- if $d(u_i) \neq d(u_j)$, then there must exist $a \in B$ such
that $a(u_i) \neq a(u_j)$, so $u_i$ and $u_j$ cannot belong to the same equivalence class $E$.

**(2.)** Suppose that $X \cap E$ contains objects with the same decision but there is also
$u \in E \setminus X$ with the same decision as objects in $X \cap E$. It would violate the
definition of decision bireduct -- there would exist a superset $X \cup \{u\} \supsetneq X$ such
that $B \Rrightarrow_{X \cup \{u\}} d$.

**(3.)** Because each $u \in X$ belongs to the support of its corresponding rule
$\bigwedge_{a \in B} (a = a(u)) \Rightarrow (d = d(u))$, we know that $X$ is a subset of the
considered union. Now, suppose that there exists $u_i \in U \setminus X$ which supports a rule in
$Rules(X, B)$. It would mean that there exists $u_j \in X$ such that $d(u_i) = d(u_j)$ and
$a(u_i) = a(u_j)$ for every $a \in B$. However, in such a case $B \Rrightarrow_X d$ could be
actually extended toward $B \Rrightarrow_{X \cup \{u_i\}} d$, so $(X, B)$ would not be a decision
bireduct.

## Remarks

This proposition shows in what sense decision bireducts can be equivalently represented by
collections of decision rules with predecessors based on attributes in $B$ and supporting sets of
objects summing up to $X$. Each bireduct yields deterministic rules when the universe is
restricted to $X$; objects in $U \setminus X$ serve as counterexamples or outliers. For worked
examples of bireduct-induced rules, see
[Golf Bireduct Rules](../examples/golf-bireduct-rules.md).
