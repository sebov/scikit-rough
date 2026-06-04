---
tags: [rst, core]
related: [definitions/decision_table.md, definitions/indiscernibility.md, notation_and_symbols.md]
---
# Approximations

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ and an attribute subset $B \subseteq A$ be
given. For a subset of objects $X \subseteq U$, the lower and upper approximations of $X$ induced by
$B$ are defined as follows:

## Lower Approximation

The lower approximation $\underline{X}_B$ is the union of all equivalence classes $[u]_B$ that are
completely contained within $X$:
$$
  \underline{X}_B = \bigcup \{[u]_B \in U/B : [u]_B \subseteq X \}
$$

## Upper Approximation

The upper approximation $\overline{X}_B$ is the union of all equivalence classes $[u]_B$ that have a
non-empty intersection with $X$: $$\overline{X}_B = \bigcup \{[u]_B \in U/B : [u]_B \cap X \neq
\emptyset \}$$
