---
id: prop-m-reduct-epsilon-bireduct-correspondence
type: proposition
status: complete
created: 2026-06-06
updated: 2026-06-06
tags: [complexity, bireducts, approximate-reducts]
requires:
  [concept-approximate-decision-reduct,
   concept-epsilon-decision-bireduct,
   concept-majority-function]
see_also:
  [prop-minimal-epsilon-bireduct-np-hard,
   concept-approximate-decision-reduct]
source: tmp/phd/thesis.tex
---

# Correspondence Between M-Reducts and Epsilon-Bireducts

There is a direct correspondence between the smallest $M$-reducts and $\varepsilon$-decision bireducts.
This connection allows transferring complexity results between the two frameworks.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ and $B \subseteq A$ be given. $B$ is the smallest $M$-reduct in
$\mathbb{A}$ if and only if there exists a subset $X \subseteq U$ such that $(X, B)$ is an
$\varepsilon$-decision bireduct and there are no other $\varepsilon$-decision bireducts with fewer
attributes than $\lvert B \rvert$.

## Proof

**($\Rightarrow$)** Suppose that $B \subseteq A$ is the smallest $M$-reduct in $\mathbb{A}$. The
majority function $M : 2^A \to [0, 1]$ is defined as a sum of fractions of the number of objects
with the most frequent decision to the number of all objects within the equivalence classes induced
by a subset of attributes. From the definition of approximate decision reduct, $M(B) \geq 1 - \varepsilon$.

If we put $X$ equal to the union of objects with the most frequent decision within each
indiscernibility class induced by $B$, then $(X, B)$ is an $\varepsilon$-decision bireduct. This is
because:

1. $B$ cannot be reduced in the context of $X$ -- otherwise it could also be reduced in the context
   of the $M$-reduct.
2. $X$ cannot be extended -- if $U \setminus X$ is not empty, then extending $X$ by any
   $u \in U \setminus X$ would violate the decision bireduct condition for $(X \cup \{u\}, B)$, as
   $u$ has a different decision than objects from $X$ that belong to the same indiscernibility class
   induced by $B$.

Let us continue by contradiction. Suppose that there exists an $\varepsilon$-decision bireduct
$(X', B')$ for which $\lvert B' \rvert < \lvert B \rvert$. Consider the indiscernibility classes
induced by $B'$ and the given $X'$. Having no knowledge of the arrangement of object belonging to
$X'$ within the indiscernibility classes, we know that $\lvert X' \rvert \geq (1 - \varepsilon)\lvert U \rvert$.
If we replace objects having not the most common decision with those having the most frequent value
within each indiscernibility class, then the coverage condition will also hold. This gives us that
$B'$ is an $M$-reduct that contains a smaller number of attributes than $B$ -- a contradiction.

**($\Leftarrow$)** Suppose that $(X, B)$ for $X \subseteq U$ and $B \subseteq A$ is an
$\varepsilon$-decision bireduct and there are no other $\varepsilon$-decision bireducts with fewer
attributes than $\lvert B \rvert$. Using the same line of reasoning as above, we know that $B$ is an
$M$-reduct in $\mathbb{A}$. However, suppose that it is not the smallest one and let $B'$ be an
$M$-reduct for which $\lvert B' \rvert < \lvert B \rvert$. Using the argumentation from the
beginning of the proof, we can show that there exists $X'$ such that $(X', B')$ is an
$\varepsilon$-decision bireduct. This leads to a contradiction, as $(X', B')$ is an
$\varepsilon$-decision bireduct with fewer attributes than in $B$.

## Remarks

This proposition establishes a bidirectional correspondence: the smallest $M$-reducts correspond to
the smallest $\varepsilon$-decision bireducts (in terms of attribute count). This allows transferring
complexity results from approximate reducts to epsilon-bireducts.

The construction in the proof shows how to convert between the two frameworks:
- From $M$-reduct to $\varepsilon$-bireduct: take objects with the most frequent decision in each class.
- From $\varepsilon$-bireduct to $M$-reduct: replace minority objects with majority objects in each class.
