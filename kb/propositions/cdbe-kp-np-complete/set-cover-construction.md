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

Given a Set Cover instance $(W, \mathcal{S})$, one can construct in polynomial time a decision
table $\mathbb{A}_{\mathcal{S}}$ whose bireducts encode the structure of the Set Cover solution.

## Statement

Given a Set Cover instance with universe $W$ and a family $\mathcal{S} = \{S_1, \dots, S_n\}$ of
subsets of $W$ such that $\bigcup \mathcal{S} = W$, define
$\mathbb{A}_{\mathcal{S}} = (U_{\mathcal{S}} \cup \{u_*\}, A_{\mathcal{S}} \cup \{d_{\mathcal{S}}\})$
as follows:

- $U_{\mathcal{S}} = \{u_\omega \mid \omega \in W\}$ --- each object corresponds to an element of
  $W$,
- $u_* \notin U_{\mathcal{S}}$ --- a special object with no counterpart in $W$,
- $A_{\mathcal{S}} = \{a_{S_i} \mid S_i \in \mathcal{S}\}$ --- each conditional attribute
  $a_{S_i} : U_{\mathcal{S}} \cup \{u_*\} \to \{0, 1\}$ is defined as

  $$
  a_{S_i}(u) =
  \begin{cases}
  1 & \text{if } u = u_\omega \text{ for some } \omega \in S_i,\\[2pt]
  0 & \text{otherwise},
  \end{cases}
  $$

- $d_{\mathcal{S}} : U_{\mathcal{S}} \cup \{u_*\} \to \{0, 1\}$ --- the decision attribute
  defined as

  $$
  d_{\mathcal{S}}(u) =
  \begin{cases}
  1 & \text{if } u = u_*,\\
  0 & \text{if } u \in U_{\mathcal{S}}.
  \end{cases}
  $$

The construction runs in $O(|W| \cdot |\mathcal{S}|)$ time.

## Proof

The construction iterates in nested loops over each $S_i \in \mathcal{S}$ and each
$\omega \in W$, putting the value of $a_{S_i}(u_\omega)$ based on the check whether
$\omega \in S_i$. Assuming set membership can be tested in $O(1)$ amortized time (e.g.,
using hash sets, requiring $O(|S_i|)$ per subset to build each hash set, which sums up to
$O(|\mathcal{S}| \cdot |W|)$), filling the $|\mathcal{S}| \cdot |W|$ entries of the final
table takes $O(|\mathcal{S}| \cdot |W|)$. Defining $u_*$ and $d_{\mathcal{S}}$ adds only
linear time. If we consider checking set membership based on the naive list-based approach,
instead of amortized hash-set-based, it still gives us $O(|\mathcal{S}| \cdot |W|^2)$.
Hence, the entire construction is polynomial with respect to the size of the Set Cover
instance.

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

Further structural properties of $\mathbb{A}_{\mathcal{S}}$ (consistency, bireduct
characterization, description length formulas) are established in subsequent lemmas.
