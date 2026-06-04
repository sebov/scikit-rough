---
id: concept-approximations
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, approximation]
requires: [concept-decision-table, concept-indiscernibility]
see_also: [concept-positive-region, concept-consistency]
source: tmp/phd/thesis.tex
---

# Approximations

Lower and upper approximations are the core mechanism through which rough set theory handles imperfect
knowledge about concepts. They approximate a target set of objects using the equivalence classes
induced by an attribute subset.

## Definition

Let $\mathbb{A} = (U, A \cup \{d\})$ and an attribute subset $B \subseteq A$ be given. For a subset
of objects $X \subseteq U$, the lower and upper approximations of $X$ induced by $B$ are:

$$
\begin{aligned}
\underline{X}_B &= \bigcup \{[u]_B \in U/B : [u]_B \subseteq X\}
  & \text{(lower approximation)} \\
\overline{X}_B &= \bigcup \{[u]_B \in U/B : [u]_B \cap X \neq \emptyset\}
  & \text{(upper approximation)}
\end{aligned}
$$

## Intuition

- **Lower approximation** $\underline{X}_B$: objects that *certainly* belong to $X$ given the
  knowledge provided by $B$. These are objects whose entire $B$-indiscernibility class is contained
  within $X$.
- **Upper approximation** $\overline{X}_B$: objects that *possibly* belong to $X$ given the knowledge
  provided by $B$. These are objects whose $B$-indiscernibility class has at least one element in
  $X$.

The difference $\overline{X}_B \setminus \underline{X}_B$ is the *boundary region* -- objects that
cannot be classified with certainty.

## Remarks

Approximations are fundamental to the [positive region](../concepts/positive-region.md) concept:
$POS(B)$ is the union of lower approximations of all decision classes induced by $B$. They also
underpin [consistency](../concepts/consistency.md) analysis of decision tables.
