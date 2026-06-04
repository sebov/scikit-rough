---
tags: [rst, core, rules]
related: [definitions/decision_table.md, notation_and_symbols.md]
---
# Decision Rules

## Formulae

Let a decision table $\mathbb{A} = (U, A \cup \{d\})$ be given. We define

$$V = \bigcup_{a \in A \cup \{d\}} V_a$$

Atomic formulae over $B \subseteq A \cup \{d\}$ and $V$ are expressions of the form $a = v$, called
**descriptors** (or **selectors**), where $a \in B$ and $v \in V_a$.

The set $\Phi(B, V)$ of formulae over $B$ and $V$ is the least set containing all atomic formulae
over $B$ and $V$ and closed with respect to the propositional operators:

- $\wedge$ (conjunction)
- $\vee$ (disjunction)
- $\neg$ (negation)

By $\|\varphi\|_\mathbb{A}$ we denote the **meaning** of $\varphi \in \Phi(B, V)$ in the decision
table $\mathbb{A}$. It is defined as the set of all objects in $U$ with the property $\varphi$:

$$\|a = v\|_\mathbb{A} = \{u \in U : a(u) = v\}$$

$$\|\varphi \wedge \varphi'\|_\mathbb{A} = \|\varphi\|_\mathbb{A} \cap \|\varphi'\|_\mathbb{A}$$

$$\|\varphi \vee \varphi'\|_\mathbb{A} = \|\varphi\|_\mathbb{A} \cup \|\varphi'\|_\mathbb{A}$$

$$\|\neg \varphi\|_\mathbb{A} = U \setminus \|\varphi\|_\mathbb{A}$$

The formulae from $\Phi(A, V)$ are called **condition formulae** of $\mathbb{A}$.

The formulae from $\Phi(\{d\}, V)$ are called **decision formulae** of $\mathbb{A}$.

## Decision Rule

A decision rule for $\mathbb{A} = (U, A \cup \{d\})$ is any expression of the form

$$\varphi \Rightarrow \psi$$

where $\varphi \in \Phi(A, V)$ and $\psi \in \Phi(\{d\}, V)$.

- $\varphi$ is called the **predecessor** of the decision rule.
- $\psi$ is called the **successor** of the decision rule.

A decision rule $\varphi \Rightarrow \psi$ is **true** in $\mathbb{A}$ if and only if
$\|\varphi\|_\mathbb{A} \subseteq \|\psi\|_\mathbb{A}$.

### Example

An exemplary decision rule has the form:

$$\underbrace{(a_1 = v_{a_1}) \wedge \dots \wedge (a_m = v_{a_m})}_{\text{predecessor}}
\;\Rightarrow\;
\underbrace{(d = v_d)}_{\text{successor}}$$

where $a_1, \dots, a_m \in A$ are conditional attributes and each $a_i = v_{a_i}$ is a descriptor.

A descriptor $a = v_a$ is **supported** by objects $u \in U$ for which $a(u) = v_a$ holds.

The rule's **support** is the set of objects matching all descriptors in both the predecessor and
the successor, i.e. the intersection of the sets supporting individual descriptors.

### Confidence

The **confidence** of a decision rule is the ratio of the cardinality of the rule's support to the
cardinality of the set of objects matching all descriptors from the predecessor:

$$
  \text{confidence} = \frac{|\text{support}_\mathbb{A}(\varphi \Rightarrow \psi)|}{|\|\varphi\|_\mathbb{A}|}
$$
