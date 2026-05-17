---
tags: [rst, core, proposition, bireducts]
related: [definitions/reducts.md, definitions/bireducts.md, definitions/decision_table.md]
---

# Decision Reduct as a Decision Bireduct

## Proposition

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ and $B \subseteq A$ be given. $B$ is a
decision reduct if and only if $(U, B)$ is a decision bireduct.

## Proof

$(\Rightarrow)$ If $B$ is a decision reduct, then $B$ is an irreducible subset of attributes that
discerns all pairs $u_i, u_j \in U$ with $d(u_i) \neq d(u_j)$. Hence $B \Rightarrow_U d$, and no
proper subset $B' \subsetneq B$ satisfies $B' \Rightarrow_U d$. Therefore $(U, B)$ is a decision
bireduct.

$(\Leftarrow)$ If $(U, B)$ is a decision bireduct, then $B \Rightarrow_U d$ and there is no proper
$B' \subsetneq B$ such that $B' \Rightarrow_U d$. Thus $B$ is a decision reduct.

## Decision Bireduct via Subtable Consistency

A decision bireduct can also be characterized in terms of a decision reduct of a restricted
subtable.

### Proposition

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. For $B \subseteq A$ and
$X \subseteq U$, denote by $\mathbb{A}[X, B]$ the decision table obtained from $\mathbb{A}$ by
removing objects outside $X$ and attributes outside $B$.

Then $(X, B)$ is a decision bireduct for $\mathbb{A}$ if and only if both of the following hold:

1. $\mathbb{A}[X, B]$ is consistent and there is no $X' \supsetneq X$ such that
   $\mathbb{A}[X', B]$ is consistent.
2. $B$ is a decision reduct for $\mathbb{A}[X, B]$.

### Proof

$(\Rightarrow)$ If $(X, B)$ is a decision bireduct, then $B \Rightarrow_X d$ and there is no proper
$B' \subsetneq B$ with $B' \Rightarrow_X d$. Hence $B$ is a decision reduct for $\mathbb{A}[X, B]$.
Consequently $\mathbb{A}[X, B]$ is consistent. Maximality of $X$ in the bireduct implies there is
no $X' \supsetneq X$ such that $\mathbb{A}[X', B]$ is consistent.

$(\Leftarrow)$ If $B$ is a decision reduct for $\mathbb{A}[X, B]$, then $B \Rightarrow_X d$ and $B$
is irreducible with respect to $X$. Since there is no $X' \supsetneq X$ with consistent
$\mathbb{A}[X', B]$, there is also no $X' \supsetneq X$ with $B \Rightarrow_{X'} d$. Thus $(X, B)$
is a decision bireduct.
