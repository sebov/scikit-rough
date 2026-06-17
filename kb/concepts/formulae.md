---
id: concept-formulae
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, rules]
requires: [concept-decision-table]
see_also: [concept-decision-rule, concept-decision-reduct]
source: src-thesis-phd
---

# Formulae

Formulae define the propositional language used to express conditions on objects in a decision table.
They are the building blocks of decision rules and Boolean characterizations of reducts.

## Definition

Let $\mathbb{A} = (U, A \cup \{d\})$ be given and let $V = \bigcup_{a \in A \cup \{d\}} V_a$.

Atomic formulae over $B \subseteq A \cup \{d\}$ and $V$ are expressions of the form $a = v$, called
**descriptors** (or **selectors**), where $a \in B$ and $v \in V_a$.

The set $\mathscr{F}(B, V)$ of formulae over $B$ and $V$ is the least set containing all atomic
formulae over $B$ and $V$ and closed under the propositional operators:

- $\land$ (conjunction)
- $\lor$ (disjunction)
- $\neg$ (negation)

### Meaning of Formulae

By $\|\varphi\|_{\mathbb{A}}$ we denote the meaning of $\varphi \in \mathscr{F}(B, V)$ in
$\mathbb{A}$ -- the set of all objects in $U$ satisfying $\varphi$:

$$
\begin{aligned}
\|a = v\|_{\mathbb{A}} &= \{u \in U : a(u) = v\} \\
\|\varphi \land \varphi'\|_{\mathbb{A}} &= \|\varphi\|_{\mathbb{A}} \cap \|\varphi'\|_{\mathbb{A}} \\
\|\varphi \lor \varphi'\|_{\mathbb{A}} &= \|\varphi\|_{\mathbb{A}} \cup \|\varphi'\|_{\mathbb{A}} \\
\|\neg \varphi\|_{\mathbb{A}} &= U \setminus \|\varphi\|_{\mathbb{A}}
\end{aligned}
$$

Formulae from $\mathscr{F}(A, V)$ are called **condition formulae**. Formulae from
$\mathscr{F}(\{d\}, V)$ are called **decision formulae**.

## Remarks

A descriptor $a = v$ is supported by objects $u \in U$ for which $a(u) = v$. This notion of support
extends to compound formulae through the semantic rules above.

Formulae are the foundation for [decision rules](../concepts/decision-rule.md) and the Boolean
characterization of [decision reducts](../concepts/reducts.md) via prime implicants.
