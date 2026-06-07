---
id: concept-approximate-decision-reduct
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, reduction, approximation]
requires:
  [concept-decision-table,
   concept-decision-reduct,
   concept-positive-region]
see_also:
  [concept-majority-function,
   concept-relative-gain-function,
   concept-discernibility-measure,
   concept-epsilon-decision-bireduct,
   prop-relative-gamma-epsilon-reduct-np-hard,
   prop-gamma-epsilon-reduct-np-hard,
   prop-relative-m-epsilon-reduct-np-hard,
   prop-m-epsilon-reduct-np-hard,
   prop-relative-r-epsilon-reduct-np-hard,
   prop-r-epsilon-reduct-np-hard]
source: tmp/phd/thesis.tex
---

# Approximate Decision Reduct

Approximate decision reducts generalize decision reducts by relaxing the requirement of perfect
discernibility. Instead, they use a threshold $\varepsilon$ and an evaluation function $F$ to allow
some loss of decision information in exchange for smaller attribute subsets.

## Evaluation Function

An evaluation function $F : 2^A \to [0, 1]$ is nondecreasing and monotone with respect to set
inclusion: $B \subseteq B' \implies F(B) \leq F(B')$. Common choices include $\gamma$ (dependency
degree), $M$ (majority function), and $R$ (relative gain function).

## Relative Approximate Decision Reduct

Let $\varepsilon \in [0, 1)$ and a monotone function $F : 2^A \to [0, 1]$ be given. A subset
$B \subseteq A$ is a relative $F$-decision $\varepsilon$-superreduct if:

$$
F(B) \geq (1 - \varepsilon)F(A)
$$

A subset $B \subseteq A$ is a relative $F$-decision $\varepsilon$-reduct if it satisfies the above
inequality and none of its proper subsets does.

## Approximate Decision Reduct (Absolute)

Let $\varepsilon \in [0, 1)$ and a monotone function $F : 2^A \to [0, 1]$ be given. A subset
$B \subseteq A$ is an $F$-decision $\varepsilon$-superreduct if:

$$
F(B) \geq 1 - \varepsilon
$$

A subset $B \subseteq A$ is an $F$-decision $\varepsilon$-reduct if it satisfies the above
inequality and none of its proper subsets does.

## Comparison of Variants

The relative version evaluates subsets in relation to the full set of attributes (threshold
$(1 - \varepsilon)F(A)$), while the absolute version uses a fixed threshold ($1 - \varepsilon$).

For consistent tables, $\gamma(A) = M(A) = R(A) = 1$, and both variants coincide whenever
$F \in \{\gamma, M, R\}$. Moreover, relative $0$-reducts for these functions are equivalent to
standard decision reducts. For inconsistent tables, the two variants may yield different subsets.

## Role of the Threshold $\varepsilon$

- **Higher $\varepsilon$**: Fewer attributes, shorter rules. By accepting slight inconsistencies we
  gain simplicity and robustness against unseen cases.
- **Lower $\varepsilon$**: More attributes, more complex rules, but higher accuracy on the training
  data.

## Complexity

Finding a minimal $F$-decision $\varepsilon$-reduct is NP-hard for each of the three main evaluation
functions. The proofs form a chain of polynomial reductions from the Minimal Dominating Set problem:

- **$\gamma$-decision $\varepsilon$-reduct**: reduction from Minimal Dominating Set, using a
  construction with $t(\varepsilon)$ auxiliary objects and $|\mathbb{V}|$ binary attributes. The
  threshold $t(\varepsilon)$ forces the positive region to cover auxiliary objects, translating
  $\gamma$-superreducts into dominating sets. Both [relative](../propositions/relative-gamma-epsilon-reduct-np-hard.md)
  and [absolute](../propositions/gamma-epsilon-reduct-np-hard.md) variants are NP-hard.
- **$M$-decision $\varepsilon$-reduct**: reduction from Minimal $\alpha(\varepsilon)$-Dominating Set
  via a construction with $m(\varepsilon)|\mathbb{V}|$ objects and an explicit formula linking
  $M(B)$ to the coverage of the induced dominating set. Both [relative](../propositions/relative-m-epsilon-reduct-np-hard.md)
  and [absolute](../propositions/m-epsilon-reduct-np-hard.md) variants are NP-hard.
- **$R$-decision $\varepsilon$-reduct**: follows from the $M$ case by observing that $R(B) = M(B)$
  for any $B$ in the same constructed table. Both [relative](../propositions/relative-r-epsilon-reduct-np-hard.md)
  and [absolute](../propositions/r-epsilon-reduct-np-hard.md) variants are NP-hard.

For consistent tables ($\gamma(A) = M(A) = R(A) = 1$), the relative and absolute conditions
coincide, so proofs for one variant carry over to the other.
