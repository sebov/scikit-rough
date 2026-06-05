---
id: prop-gamma-decision-reduct-boolean-formula
type: proposition
status: complete
created: 2026-06-05
updated: 2026-06-05
tags: [core, reduction, positive-region]
requires:
  [concept-decision-table,
   concept-positive-region,
   concept-gamma-decision-reduct]
see_also:
  [prop-decision-reduct-boolean-formula,
   prop-gamma-decision-reduct-consistent-table]
source: tmp/phd/thesis.tex
---

# Gamma-Decision Reduct Boolean Formula Characterisation

$\gamma$-decision reducts correspond to prime implicants of a modified Boolean formula that restricts
discernibility checks to object pairs where the first element belongs to the positive region.

## Statement

Let $\mathbb{A} = (U, A \cup \{d\})$ be given. Consider the following Boolean formula with
propositional variables $\overline{a}$ for $a \in A$:

$$
\tau^\gamma = \bigwedge_{u_i \in POS(A)}
              \bigwedge_{\substack{u_j \in U \\ d(u_i) \neq d(u_j)}}
              \bigvee_{\substack{a \in A \\ a(u_i) \neq a(u_j)}}
              \overline{a}
$$

A subset $B \subseteq A$ is a $\gamma$-decision reduct if and only if the Boolean formula
$\bigwedge_{a \in B} \overline{a}$ is a prime implicant for $\tau^\gamma$.

## Proof

See Skowron and Rauszer (1992), "The Discernibility Matrices and Functions in Information Systems".

The construction follows the same Boolean reasoning framework as the standard decision reduct
characterisation (prop-decision-reduct-boolean-formula). The key difference is that clauses are
generated only for pairs $(u_i, u_j)$ where $u_i \in POS(A)$ and $d(u_i) \neq d(u_j)$. Objects
outside the positive region do not contribute clauses, reflecting the fact that $\gamma$-decision
reducts only need to preserve the discernibility of deterministically classifiable objects.

## Consequences

This result enables computation of $\gamma$-decision reducts via the same Boolean minimisation
algorithms used for standard reducts. The formula $\tau^\gamma$ is typically simpler than $\tau$
(fewer conjuncts), since pairs involving objects outside $POS(A)$ are excluded.
