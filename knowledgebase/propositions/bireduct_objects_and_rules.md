---
tags: [rst, core, proposition, bireducts, rules]
related: [definitions/bireducts.md, definitions/decision_rules.md, definitions/decision_table.md]
---

# Bireduct Objects and Rules

## Proposition

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given and $(X, B)$ be a decision bireduct
for $\mathbb{A}$. The following statements are true:

1.  For each equivalence class $E \in U / B$, all objects in $X \cap E$ have the same decision
    value, further denoted as $v_{d(X \cap E)} \in V_d$.

2.  For each equivalence class $E \in U / B$, all objects in $E$ which have value $v_{d(X \cap E)}
    \in V_d$ on $d$ are contained in $X$.

3.  $X$ equals the union of supports of the following set of decision rules:

    $$
    Rules(X, B) =
    \left\{
        \bigwedge_{a \in B} (a = a(u)) \Rightarrow (d = d(u)) : u \in X
    \right\}
    $$

## Proof

**(1.)** Suppose that $X \cap E$ contains objects from more than one decision class, i.e., there are
objects $u_i, u_j \in X \cap E$ such that $d(u_i) \neq d(u_j)$. It would violate the definition of a
decision bireduct -- if $d(u_i) \neq d(u_j)$, then there must exist $a \in B$ such that $a(u_i) \neq
a(u_j)$, so $u_i$ and $u_j$ cannot belong to the same equivalence class $E$.

**(2.)** Suppose that $X \cap E$ contains objects with the same decision but there is also $u \in E
\setminus X$ with the same decision as objects in $X \cap E$. It would violate the definition of a
decision bireduct -- there would exist a superset $X \cup \{u\} \supsetneq X$ such that $B
\Rightarrow_{X \cup \{u\}} d$.

**(3.)** Because each $u \in X$ belongs to the support of its corresponding rule $\bigwedge_{a \in B}
(a = a(u)) \Rightarrow (d = d(u))$, we know that $X$ is a subset of the considered union. Now,
suppose that there exists $u_i \in U \setminus X$ which supports a rule in $Rules(X, B)$. It would
mean that there exists $u_j \in X$ such that $d(u_i) = d(u_j)$ and $a(u_i) = a(u_j)$ for every $a
\in B$. However, in such a case $B \Rightarrow_X d$ could be extended toward $B \Rightarrow_{X \cup
\{u_i\}} d$, so $(X, B)$ would not be a decision bireduct.
