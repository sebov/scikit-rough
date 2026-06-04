---
id: ex-golf-all-bireducts
type: example
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [bireducts, decision-table]
requires:
  [concept-decision-table,
   concept-decision-bireduct,
   concept-gamma-decision-bireduct]
see_also: [ex-golf-bireduct-rules]
source: tmp/phd/thesis.tex
---

# Golf Dataset -- Complete Bireduct Listing

A complete list of all decision bireducts and $\gamma$-decision bireducts for the golf dataset
($\lvert U \rvert = 14$, $A = \{\text{Outlook}, \text{Temperature}, \text{Humidity}, \text{Wind}\}$).
Each row shows an attribute subset $B \subseteq A$ and all valid covered object subsets $X \subseteq U$
such that $(X, B)$ is a bireduct.

Entries marked **NA** indicate that the attribute subset can be reduced (irreducibility fails,
violating condition 2 of the bireduct definition). An empty set $\emptyset$ in the gamma column means
$POS(B) = \emptyset$ (no objects belong to the positive region).

| $B \subseteq A$  | Decision bireducts: $X$                                                                                                                                                                                                                                                                           | $\gamma$-bireducts: $X$                                       |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------ |
| $\emptyset$      | $\{u_1, u_2, u_6, u_8, u_{14}\}$; $\{u_3, u_4, u_5, u_7, u_9, u_{10}, u_{11}, u_{12}, u_{13}\}$                                                                                                                                                                                                   | $\emptyset$                                                   |
| $\{O\}$          | $\{u_1, \ldots, u_5, u_7, u_8, u_{10}, u_{12}, u_{13}\}$; $\{u_1, u_2, u_3, u_6, u_7, u_8, u_{12}, u_{13}, u_{14}\}$; $\{u_3, u_6, u_7, u_9, u_{11}, u_{12}, u_{13}, u_{14}\}$                                                                                                                    | $\{u_3, u_7, u_{12}, u_{13}\}$                                |
| $\{T\}$          | $\{u_1, u_2, u_4, u_5, u_7, u_9, u_{10}, u_{11}, u_{12}\}$; $\{u_1, u_2, u_4, u_6, u_{10}, u_{11}, u_{12}\}$; $\{u_1, u_2, u_5, u_7, u_8, u_9, u_{14}\}$; $\{u_3, u_4, u_6, u_{10}, u_{11}, u_{12}, u_{13}\}$; $\{u_3, u_5, u_7, u_8, u_9, u_{13}, u_{14}\}$; $\{u_3, u_6, u_8, u_{13}, u_{14}\}$ | NA                                                            |
| $\{H\}$          | $\{u_1, u_2, u_5, u_7, \ldots, u_{11}, u_{13}, u_{14}\}$; $\{u_3, u_4, u_6, u_{12}\}$                                                                                                                                                                                                             | NA                                                            |
| $\{W\}$          | $\{u_1, u_7, u_8, u_{11}, u_{12}\}$; $\{u_2, \ldots, u_6, u_9, u_{10}, u_{13}, u_{14}\}$                                                                                                                                                                                                          | NA                                                            |
| $\{O, T\}$       | 8 subsets (range 8--13 objects)                                                                                                                                                                                                                                                                   | $\{u_1, u_2, u_3, u_7, u_9, u_{12}, u_{13}\}$                 |
| $\{O, H\}$       | 4 subsets (range 9--11 objects)                                                                                                                                                                                                                                                                   | $\{u_1, u_2, u_3, u_7, u_8, u_9, u_{11}, u_{12}, u_{13}\}$    |
| $\{O, W\}$       | 4 subsets (range 10--11 objects)                                                                                                                                                                                                                                                                  | $\{u_3, u_4, u_5, u_6, u_7, u_{10}, u_{12}, u_{13}, u_{14}\}$ |
| $\{T, H\}$       | 5 subsets (range 8--10 objects)                                                                                                                                                                                                                                                                   | $\{u_{10}, u_{11}, u_{13}\}$                                  |
| $\{T, W\}$       | 13 subsets (range 8--11 objects)                                                                                                                                                                                                                                                                  | $\{u_2, u_5, u_9\}$                                           |
| $\{H, W\}$       | 5 subsets (range 9--11 objects)                                                                                                                                                                                                                                                                   | $\{u_5, u_9, u_{10}, u_{13}\}$                                |
| $\{O, T, H\}$    | 2 subsets (range 10--11 objects)                                                                                                                                                                                                                                                                  | $\{u_1, u_2, u_3, u_7, \ldots, u_{13}\}$                      |
| $\{O, T, W\}$    | $U$ (all 14 objects)                                                                                                                                                                                                                                                                              | $U$                                                           |
| $\{O, H, W\}$    | $U$ (all 14 objects)                                                                                                                                                                                                                                                                              | $U$                                                           |
| $\{T, H, W\}$    | 11 subsets (range 10--12 objects)                                                                                                                                                                                                                                                                 | $\{u_2, u_5, u_9, u_{10}, u_{11}, u_{13}\}$                   |
| $\{O, T, H, W\}$ | NA                                                                                                                                                                                                                                                                                                | NA                                                            |

## Remarks

Key observations from the complete listing:

- The same $B \subseteq A$ can occur as a component of **many** decision bireducts with different
  $X$, but as a component of at most **one** $\gamma$-decision bireduct (since $X = POS(B)$ is
  uniquely determined).
- For the same $B$, the corresponding $X$ is usually **larger** for decision bireducts than for
  $\gamma$-decision bireducts, because the $\gamma$ variant imposes stricter discernibility
  conditions (against all of $U$, not just $X$).
- Decision bireducts are substantially more numerous than standard decision reducts (2 reducts vs
  dozens of bireducts).
- Singleton attribute subsets can form bireducts (e.g., $\{O\}$ yields 3 decision bireducts).
- The empty attribute set $\emptyset$ yields 2 bireducts, each covering a single decision class
  entirely. This corresponds to "dummy" classifiers always predicting one class.
- The two decision reducts $\{O, T, W\}$ and $\{O, H, W\}$ appear as the only $B$ with $X = U$
  (full coverage).
- $\{O, T, H, W\}$ yields NA in both columns because the full attribute set is redundant -- proper
  subsets already discern all objects.
