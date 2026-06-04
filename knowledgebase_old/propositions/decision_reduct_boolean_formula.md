---
tags: [rst, core, proposition]
related: [notation_and_symbols.md, definitions/reducts.md, definitions/decision_table.md, definitions/consistency.md]
---
# Decision Reduct Boolean Formula

## Proposition

Let a consistent decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. Consider the
following Boolean formula with propositional variables $\overline{a}$ for $a \in A$:

$$
  \tau =
    \bigwedge_{\substack{u_i, u_j \in U \\ i < j, \; d(u_i) \neq d(u_j)}}
    \bigvee_{\substack{a \in A \\ a(u_i) \neq a(u_j)}}
    \overline{a}
$$

An arbitrary subset $B \subseteq A$ is a decision reduct if and only if the Boolean formula
$\bigwedge_{a \in B} \overline{a}$ is a prime implicant for $\tau$.
