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
   concept-epsilon-decision-bireduct]
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

Finding a minimal $F$-decision $\varepsilon$-reduct is NP-hard for functions including $\gamma$, $M$,
and $R$. Proofs typically reduce from the Minimal Dominating Set problem.
