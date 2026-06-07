---
id: concept-decision-rule
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, rules]
requires: [concept-decision-table, concept-formulae]
see_also: [concept-decision-reduct, concept-decision-bireduct]
source: src-thesis-phd
---

# Decision Rule

A decision rule is an if-then expression that describes the relationship between conditional
attributes and the decision attribute. Collections of decision rules form the basis of rule-based
classifiers in rough set theory.

## Definition

Let $\mathbb{A} = (U, A \cup \{d\})$ be given and let $V = \bigcup_{a \in A \cup \{d\}} V_a$.

A decision rule for $\mathbb{A}$ is any expression of the form:

$$
\varphi \Rightarrow \psi
$$

where $\varphi \in \mathscr{F}(A, V)$ and $\psi \in \mathscr{F}(\{d\}, V)$.

- $\varphi$ is called the **predecessor** (or antecedent) of the rule.
- $\psi$ is called the **successor** (or consequent) of the rule.

A decision rule $\varphi \Rightarrow \psi$ is **true** in $\mathbb{A}$ if and only if:

$$
\|\varphi\|_{\mathbb{A}} \subseteq \|\psi\|_{\mathbb{A}}
$$

## Structure

An exemplary decision rule takes the form:

$$
\underbrace{(a_1 = v_{a_1}) \land \cdots \land (a_m = v_{a_m})}_{\text{predecessor}}
\;\Rightarrow\;
\underbrace{(d = v_d)}_{\text{successor}}
$$

where $a_1, \ldots, a_m \in A$ are conditional attributes. Each $a_i = v_{a_i}$ is a descriptor
(atomic formula).

## Support and Confidence

The **support** of a rule $\varphi \Rightarrow \psi$ is the set of objects matching all descriptors
in both the predecessor and successor, i.e., $\|\varphi \land \psi\|_{\mathbb{A}}$.

The **confidence** of a rule is the ratio:

$$
\text{confidence} = \frac{\lvert \|\varphi \land \psi\|_{\mathbb{A}} \rvert}
                            {\lvert \|\varphi\|_{\mathbb{A}} \rvert}
$$

## Remarks

Decision rules generated from [decision reducts](../concepts/decision-reduct.md) have confidence
equal to 1 (they are deterministic). Rules generated from [decision
bireducts](../concepts/decision-bireduct.md) are deterministic within the covered object subset $X$
but may have counterexamples in $U \setminus X$.
