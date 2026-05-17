---
tags: [rst, core, proposition]
related: [notation_and_symbols.md, definitions/reducts.md, definitions/positive_region.md, definitions/decision_table.md, definitions/consistency.md]
---

# Gamma-Decision Reduct Characterization

## Background

The transformation of an inconsistent decision table to a consistent counterpart via an appropriate
replacement of the decision attribute is a well-known mechanism in rough set literature. The
interrelations between different decision reduct (or superreduct) variants have been studied
extensively, allowing formal expression of the intuitions behind gamma-decision reducts.

## Proposition

Let an inconsistent decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. An arbitrary subset
$B \subseteq A$ is a $\gamma$-decision reduct in $\mathbb{A}$ if and only if $B$ is a decision
reduct in $\mathbb{A}_A^\gamma$, where $d_A^\gamma$ is defined as:

$$
  d_A^\gamma(u) =
  \begin{cases}
    *      & \text{if } u \notin POS_A(d), \\
    d(u)   & \text{otherwise}.
  \end{cases}
$$

## Boolean Formula Characterization

An analogous result to the standard decision reduct Boolean formula can be formulated for
$\gamma$-decision reducts. The key difference is that only objects from the positive region
$POS_A(d)$ are considered as the first component of the discerned pairs.

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. Consider the following Boolean
formula $\Phi^\gamma$ with propositional variables $\overline{a}$ for $a \in A$:

$$
  \Phi^\gamma =
    \bigwedge_{u_i \in POS_A(d)}
    \bigwedge_{\substack{u_j \in U \\ d(u_i) \neq d(u_j)}}
    \bigvee_{\substack{a \in A \\ a(u_i) \neq a(u_j)}}
    \overline{a}
$$

An arbitrary subset $B \subseteq A$ is a $\gamma$-decision reduct if and only if the Boolean formula
$\bigwedge_{a \in B} \overline{a}$ is a prime implicant for $\Phi^\gamma$.
