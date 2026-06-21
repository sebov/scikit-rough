---
id: prop-set-cover-construction
type: proposition
status: draft
created: 2026-06-21
updated: 2026-06-21
tags: [complexity, bireducts, ensemble]
requires:
  - prop-set-cover-problem
  - concept-decision-table
see_also:
  - prop-cdbe-kp-np-complete
  - concept-bireduct-ensemble
source: src-reduct-matrix-ensembles
---

# Construction of Decision Table from a Set Cover Instance

Given a Set Cover instance $(W, \mathcal{S})$, we construct in polynomial time a decision
table $\mathbb{A}_{\mathcal{S}}$ whose bireducts encode the structure of the Set Cover
solution.

## Statement

There exists a polynomial-time transformation that maps every Set Cover instance $(W, \mathcal{S})$
to a decision table $\mathbb{A}_{\mathcal{S}} = (U_{\mathcal{S}} \cup \{u_*\},
A_{\mathcal{S}} \cup \{d_{\mathcal{S}}\})$ with the following properties:

1. The universe consists of objects $u_\omega$ corresponding one-to-one to elements $\omega \in W$,
   plus a single special object $u_* \notin \{\,u_\omega \mid \omega \in W\,\}$.
2. The conditional attributes correspond one-to-one to subsets in $\mathcal{S}$.
3. The decision attribute distinguishes $u_*$ (value $1$) from all other objects (value $0$).

## Construction

Let $W$ be the Set Cover universe and $\mathcal{S} = \{S_1, \dots, S_n\}$ the family of subsets.

**Objects.** Define $U_{\mathcal{S}} = \{u_\omega \mid \omega \in W\}$ and add a special object
$u_* \notin U_{\mathcal{S}}$. The full object set is $U_{\mathcal{S}} \cup \{u_*\}$.

**Conditional attributes.** For each subset $S_i \in \mathcal{S}$, introduce a binary conditional
attribute $a_{S_i} : U_{\mathcal{S}} \cup \{u_*\} \to \{0, 1\}$ defined as

$$
a_{S_i}(u) =
\begin{cases}
1 & \text{if } u = u_\omega \text{ for some } \omega \in S_i,\\[2pt]
0 & \text{otherwise.}
\end{cases}
$$

In particular, $a_{S_i}(u_*) = 0$ for every $S_i \in \mathcal{S}$. The set of conditional
attributes is $A_{\mathcal{S}} = \{a_{S_i} \mid S_i \in \mathcal{S}\}$.

**Decision attribute.** Define $d_{\mathcal{S}} : U_{\mathcal{S}} \cup \{u_*\} \to \{0, 1\}$ as

$$
d_{\mathcal{S}}(u) =
\begin{cases}
1 & \text{if } u = u_*,\\
0 & \text{if } u \in U_{\mathcal{S}}.
\end{cases}
$$

The decision attribute is not a member of $A_{\mathcal{S}}$.

**Decision table.**

$$
\mathbb{A}_{\mathcal{S}} = \bigl(
    U_{\mathcal{S}} \cup \{u_*\},\;
    A_{\mathcal{S}} \cup \{d_{\mathcal{S}}\}
\bigr)
$$

The construction runs in time $O(|W| \cdot |\mathcal{S}|)$, which is polynomial in the size of the
Set Cover instance.

## Example

Let $W = \{\alpha, \beta, \gamma, \delta\}$ and

$$\mathcal{S} = \{\{\alpha, \beta\},\; \{\gamma, \delta\},\; \{\alpha, \delta\},\; \{\beta\}\}.$$

The resulting decision table $\mathbb{A}_{\mathcal{S}}$ is:

|                   | $a_{\{\alpha,\beta\}}$ | $a_{\{\gamma,\delta\}}$ | $a_{\{\alpha,\delta\}}$ | $a_{\{\beta\}}$ | $d_{\mathcal{S}}$ |
| :---------------- | :--------------------: | :---------------------: | :---------------------: | :-------------: | :---------------: |
| $u_\alpha$        | $1$                    | $0$                     | $1$                     | $0$             | $0$               |
| $u_\beta$         | $1$                    | $0$                     | $0$                     | $1$             | $0$               |
| $u_\gamma$        | $0$                    | $1$                     | $0$                     | $0$             | $0$               |
| $u_\delta$        | $0$                    | $1$                     | $1$                     | $0$             | $0$               |
| $u_*$             | $0$                    | $0$                     | $0$                     | $0$             | $1$               |

## Remarks

Every object in $U_{\mathcal{S}}$ has decision value $0$ and, because $\bigcup \mathcal{S} = W$,
at least one conditional attribute with value $1$. The special object $u_*$ is the unique object
with decision value $1$ and it has $0$ on every conditional attribute. Further structural
properties (consistency, bireduct characterization, description length formulas) are established
in subsequent lemmas.
