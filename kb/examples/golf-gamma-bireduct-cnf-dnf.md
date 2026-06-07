---
id: ex-golf-gamma-bireduct-cnf-dnf
type: example
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [example, golf, bireducts, boolean-formula]
requires:
  [concept-decision-table,
   concept-gamma-decision-bireduct,
   prop-gamma-decision-bireduct-boolean-formula]
see_also:
  [prop-gamma-decision-bireduct-boolean-formula,
   ex-golf-all-bireducts,
   ex-golf-bireduct-cnf-dnf]
source: tmp/phd/include/gamma_decision_bireducts_cnf_dnf.tex
---

# Golf Dataset -- Gamma-Decision Bireduct Boolean Formula (CNF/DNF)

The Boolean formula $\tau_{bi}^{\gamma}$ whose prime implicants correspond to
$\gamma$-decision bireducts of the golf dataset. Propositional variables
$\overline{u_1},\ldots,\overline{u_{14}}$ represent objects,
$\overline{O},\overline{T},\overline{H},\overline{W}$ represent attributes.

## CNF

$\tau_{bi}^{\gamma}$ is the conjunction of the following 82 clauses:

\[
\begin{aligned}
&(\overline{u_1} \vee \overline{O}) \wedge \\
&(\overline{u_1} \vee \overline{O} \vee \overline{T}) \wedge \\
&(\overline{u_1} \vee \overline{O} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_1} \vee \overline{O} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_1} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_1} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_1} \vee \overline{O} \vee \overline{T} \vee \overline{W}) \wedge \\
&(\overline{u_1} \vee \overline{O} \vee \overline{H}) \wedge \\
&(\overline{u_2} \vee \overline{O} \vee \overline{W}) \wedge \\
&(\overline{u_2} \vee \overline{O} \vee \overline{T} \vee \overline{W}) \wedge \\
&(\overline{u_2} \vee \overline{O} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_2} \vee \overline{O} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_2} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_2} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_2} \vee \overline{O} \vee \overline{T}) \wedge \\
&(\overline{u_2} \vee \overline{O} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_3} \vee \overline{O}) \wedge \\
&(\overline{u_3} \vee \overline{O} \vee \overline{W}) \wedge \\
&(\overline{u_3} \vee \overline{O} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_3} \vee \overline{O} \vee \overline{T}) \wedge \\
&(\overline{u_3} \vee \overline{O} \vee \overline{T} \vee \overline{W}) \wedge \\
&(\overline{u_4} \vee \overline{O} \vee \overline{T}) \wedge \\
&(\overline{u_4} \vee \overline{O} \vee \overline{T} \vee \overline{W}) \wedge \\
&(\overline{u_4} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_4} \vee \overline{O}) \wedge \\
&(\overline{u_4} \vee \overline{W}) \wedge \\
&(\overline{u_5} \vee \overline{O} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_5} \vee \overline{O} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_5} \vee \overline{W}) \wedge \\
&(\overline{u_5} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_6} \vee \overline{O} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_6} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_6} \vee \overline{W}) \wedge \\
&(\overline{u_6} \vee \overline{O}) \wedge \\
&(\overline{u_6} \vee \overline{O} \vee \overline{W}) \wedge \\
&(\overline{u_6} \vee \overline{T} \vee \overline{W}) \wedge \\
&(\overline{u_6} \vee \overline{O} \vee \overline{T}) \wedge \\
&(\overline{u_6} \vee \overline{O} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_6} \vee \overline{O} \vee \overline{T} \vee \overline{W}) \wedge \\
&(\overline{u_7} \vee \overline{O} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_7} \vee \overline{O} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_7} \vee \overline{O}) \wedge \\
&(\overline{u_8} \vee \overline{O} \vee \overline{T}) \wedge \\
&(\overline{u_8} \vee \overline{O}) \wedge \\
&(\overline{u_8} \vee \overline{O} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_8} \vee \overline{O} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_8} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_8} \vee \overline{O} \vee \overline{H}) \wedge \\
&(\overline{u_8} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_8} \vee \overline{O} \vee \overline{W}) \wedge \\
&(\overline{u_9} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_9} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_9} \vee \overline{O} \vee \overline{W}) \wedge \\
&(\overline{u_9} \vee \overline{O} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_{10}} \vee \overline{O} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_{10}} \vee \overline{O} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_{10}} \vee \overline{T} \vee \overline{W}) \wedge \\
&(\overline{u_{10}} \vee \overline{O} \vee \overline{H}) \wedge \\
&(\overline{u_{10}} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_{11}} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_{11}} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_{11}} \vee \overline{O} \vee \overline{T}) \wedge \\
&(\overline{u_{11}} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_{11}} \vee \overline{O} \vee \overline{H}) \wedge \\
&(\overline{u_{12}} \vee \overline{O} \vee \overline{T} \vee \overline{W}) \wedge \\
&(\overline{u_{12}} \vee \overline{O} \vee \overline{T}) \wedge \\
&(\overline{u_{12}} \vee \overline{O} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_{12}} \vee \overline{O} \vee \overline{W}) \wedge \\
&(\overline{u_{12}} \vee \overline{O}) \wedge \\
&(\overline{u_{13}} \vee \overline{O} \vee \overline{H}) \wedge \\
&(\overline{u_{13}} \vee \overline{O} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_{13}} \vee \overline{O} \vee \overline{T} \vee \overline{W}) \wedge \\
&(\overline{u_{13}} \vee \overline{O} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_{13}} \vee \overline{O} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_{14}} \vee \overline{O} \vee \overline{T} \vee \overline{W}) \wedge \\
&(\overline{u_{14}} \vee \overline{W}) \wedge \\
&(\overline{u_{14}} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_{14}} \vee \overline{O} \vee \overline{T} \vee \overline{H}) \wedge \\
&(\overline{u_{14}} \vee \overline{O} \vee \overline{T} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_{14}} \vee \overline{H} \vee \overline{W}) \wedge \\
&(\overline{u_{14}} \vee \overline{O} \vee \overline{H}) \wedge \\
&(\overline{u_{14}} \vee \overline{O})
\end{aligned}
\]

## DNF

$\tau_{bi}^{\gamma}$ is the disjunction of the following 12 terms:

\[
\begin{aligned}
&(\overline{u_1} \wedge \overline{u_2} \wedge \overline{u_3} \wedge \overline{u_4} \wedge \overline{u_5} \wedge \overline{u_6} \wedge \overline{u_7} \wedge \overline{u_8} \wedge \overline{u_9} \wedge \overline{u_{10}} \wedge \overline{u_{11}} \wedge \overline{u_{12}} \wedge \overline{u_{13}} \wedge \overline{u_{14}}) \vee \\
&(\overline{H} \wedge \overline{W} \wedge \overline{u_1} \wedge \overline{u_2} \wedge \overline{u_3} \wedge \overline{u_4} \wedge \overline{u_6} \wedge \overline{u_7} \wedge \overline{u_8} \wedge \overline{u_{11}} \wedge \overline{u_{12}} \wedge \overline{u_{14}}) \vee \\
&(\overline{O} \wedge \overline{u_1} \wedge \overline{u_2} \wedge \overline{u_4} \wedge \overline{u_5} \wedge \overline{u_6} \wedge \overline{u_8} \wedge \overline{u_9} \wedge \overline{u_{10}} \wedge \overline{u_{11}} \wedge \overline{u_{14}}) \vee \\
&(\overline{O} \wedge \overline{H} \wedge \overline{u_4} \wedge \overline{u_5} \wedge \overline{u_6} \wedge \overline{u_{10}} \wedge \overline{u_{14}}) \vee \\
&(\overline{O} \wedge \overline{H} \wedge \overline{W}) \vee \\
&(\overline{O} \wedge \overline{T} \wedge \overline{u_4} \wedge \overline{u_5} \wedge \overline{u_6} \wedge \overline{u_8} \wedge \overline{u_{10}} \wedge \overline{u_{11}} \wedge \overline{u_{14}}) \vee \\
&(\overline{O} \wedge \overline{T} \wedge \overline{H} \wedge \overline{u_4} \wedge \overline{u_5} \wedge \overline{u_6} \wedge \overline{u_{14}}) \vee \\
&(\overline{O} \wedge \overline{T} \wedge \overline{W}) \vee \\
&(\overline{O} \wedge \overline{W} \wedge \overline{u_1} \wedge \overline{u_2} \wedge \overline{u_8} \wedge \overline{u_9} \wedge \overline{u_{11}}) \vee \\
&(\overline{T} \wedge \overline{H} \wedge \overline{u_1} \wedge \overline{u_2} \wedge \overline{u_3} \wedge \overline{u_4} \wedge \overline{u_5} \wedge \overline{u_6} \wedge \overline{u_7} \wedge \overline{u_8} \wedge \overline{u_9} \wedge \overline{u_{12}} \wedge \overline{u_{14}}) \vee \\
&(\overline{T} \wedge \overline{H} \wedge \overline{W} \wedge \overline{u_1} \wedge \overline{u_3} \wedge \overline{u_4} \wedge \overline{u_6} \wedge \overline{u_7} \wedge \overline{u_8} \wedge \overline{u_{12}} \wedge \overline{u_{14}}) \vee \\
&(\overline{T} \wedge \overline{W} \wedge \overline{u_1} \wedge \overline{u_3} \wedge \overline{u_4} \wedge \overline{u_6} \wedge \overline{u_7} \wedge \overline{u_8} \wedge \overline{u_{10}} \wedge \overline{u_{11}} \wedge \overline{u_{12}} \wedge \overline{u_{13}} \wedge \overline{u_{14}})
\end{aligned}
\]

## Remarks

The prime implicants of the CNF (equivalently, prime implicates of the DNF) of
$\tau_{bi}^{\gamma}$ correspond to $\gamma$-decision bireducts of the golf dataset.
