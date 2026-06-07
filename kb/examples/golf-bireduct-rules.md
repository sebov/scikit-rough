---
id: ex-golf-bireduct-rules
type: example
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [bireducts, rules, decision-table]
requires:
  [concept-decision-table,
   concept-decision-bireduct,
   concept-gamma-decision-bireduct,
   concept-decision-rule]
see_also: [ex-golf-reduct-rules, ex-golf-all-bireducts]
source: src-thesis-phd
---

# Golf Dataset -- Bireduct Rules

Decision rules generated from sample decision bireducts and $\gamma$-decision bireducts for the golf
dataset. Rules from bireducts are deterministic within the covered object set $X$ but may have
counterexamples in $U \setminus X$.

## Decision Bireduct $(X, B) = (\{u_1, \ldots, u_3, u_6, \ldots, u_9, u_{11}, \ldots, u_{14}\}, \{\text{Outlook}, \text{Humidity}\})$

Covered objects: $X = \{u_1, u_2, u_3, u_6, u_7, u_8, u_9, u_{11}, u_{12}, u_{13}, u_{14}\}$.
Uncovered: $U \setminus X = \{u_4, u_5, u_{10}\}$.

| Rule                                                                           |    Support    |
| :----------------------------------------------------------------------------- | :-----------: |
| $(O = \text{overcast}) \land (H = \text{high}) \Rightarrow (d = \text{yes})$   |  $\{3, 12\}$  |
| $(O = \text{overcast}) \land (H = \text{normal}) \Rightarrow (d = \text{yes})$ |  $\{7, 13\}$  |
| $(O = \text{rain}) \land (H = \text{high}) \Rightarrow (d = \text{no})$        |   $\{14\}$    |
| $(O = \text{rain}) \land (H = \text{normal}) \Rightarrow (d = \text{no})$      |    $\{6\}$    |
| $(O = \text{sunny}) \land (H = \text{high}) \Rightarrow (d = \text{no})$       | $\{1, 2, 8\}$ |
| $(O = \text{sunny}) \land (H = \text{normal}) \Rightarrow (d = \text{yes})$    |  $\{9, 11\}$  |

All rules are deterministic on $X$. Objects $u_4, u_5, u_{10}$ are uncovered because they conflict
with the rule $(O = \text{rain}) \land (H = \text{normal}) \Rightarrow (d = \text{no})$ (these
objects have the same attribute values but different decision).

## Decision Bireduct $(X, B) = (\{u_1, \ldots, u_4, u_6, \ldots, u_{10}, u_{12}, u_{13}\}, \{\text{Outlook}, \text{Temperature}\})$

Covered: $X = \{u_1, u_2, u_3, u_4, u_6, u_7, u_8, u_9, u_{10}, u_{12}, u_{13}\}$.
Uncovered: $\{u_5, u_{11}, u_{14}\}$.

| Rule                                                                         |   Support   |
| :--------------------------------------------------------------------------- | :---------: |
| $(O = \text{overcast}) \land (T = \text{cool}) \Rightarrow (d = \text{yes})$ |   $\{7\}$   |
| $(O = \text{overcast}) \land (T = \text{hot}) \Rightarrow (d = \text{yes})$  | $\{3, 13\}$ |
| $(O = \text{overcast}) \land (T = \text{mild}) \Rightarrow (d = \text{yes})$ |  $\{12\}$   |
| $(O = \text{rain}) \land (T = \text{cool}) \Rightarrow (d = \text{no})$      |   $\{6\}$   |
| $(O = \text{rain}) \land (T = \text{mild}) \Rightarrow (d = \text{yes})$     | $\{4, 10\}$ |
| $(O = \text{sunny}) \land (T = \text{cool}) \Rightarrow (d = \text{yes})$    |   $\{9\}$   |
| $(O = \text{sunny}) \land (T = \text{hot}) \Rightarrow (d = \text{no})$      | $\{1, 2\}$  |
| $(O = \text{sunny}) \land (T = \text{mild}) \Rightarrow (d = \text{no})$     |   $\{8\}$   |

## $\gamma$-Decision Bireduct $(X, B) = (\{u_1, u_2, u_3, u_7, u_8, u_9, u_{11}, u_{12}, u_{13}\}, \{\text{Outlook}, \text{Humidity}\})$

Covered: $X = \{u_1, u_2, u_3, u_7, u_8, u_9, u_{11}, u_{12}, u_{13}\}$.

| Rule                                                                           |    Support    |
| :----------------------------------------------------------------------------- | :-----------: |
| $(O = \text{overcast}) \land (H = \text{high}) \Rightarrow (d = \text{yes})$   |  $\{3, 12\}$  |
| $(O = \text{overcast}) \land (H = \text{normal}) \Rightarrow (d = \text{yes})$ |  $\{7, 13\}$  |
| $(O = \text{sunny}) \land (H = \text{high}) \Rightarrow (d = \text{no})$       | $\{1, 2, 8\}$ |
| $(O = \text{sunny}) \land (H = \text{normal}) \Rightarrow (d = \text{yes})$    |  $\{9, 11\}$  |

Note that $X = POS(\{\text{Outlook}, \text{Humidity}\})$. Objects with conflicting decisions within
their equivalence classes (e.g., $u_6$ conflicts with $u_4, u_5, u_{10}$ in the
$\{\text{Outlook}, \text{Humidity}\}$-induced class) are excluded. All rules are deterministic
with respect to the entire $U$ (confidence = 1).

## $\gamma$-Decision Bireduct $(X, B) = (\{u_1, u_2, u_3, u_7, u_9, u_{12}, u_{13}\}, \{\text{Outlook}, \text{Temperature}\})$

Covered: $X = \{u_1, u_2, u_3, u_7, u_9, u_{12}, u_{13}\}$.

| Rule                                                                         |   Support   |
| :--------------------------------------------------------------------------- | :---------: |
| $(O = \text{overcast}) \land (T = \text{cool}) \Rightarrow (d = \text{yes})$ |   $\{7\}$   |
| $(O = \text{overcast}) \land (T = \text{hot}) \Rightarrow (d = \text{yes})$  | $\{3, 13\}$ |
| $(O = \text{overcast}) \land (T = \text{mild}) \Rightarrow (d = \text{yes})$ |  $\{12\}$   |
| $(O = \text{sunny}) \land (T = \text{cool}) \Rightarrow (d = \text{yes})$    |   $\{9\}$   |
| $(O = \text{sunny}) \land (T = \text{hot}) \Rightarrow (d = \text{no})$      | $\{1, 2\}$  |

## Remarks

Comparing the decision bireduct and $\gamma$-decision bireduct for the same attribute subset
$\{\text{Outlook}, \text{Humidity}\}$: the standard bireduct covers 11 objects while the
$\gamma$-bireduct covers only 9, because the $\gamma$ variant imposes the stronger requirement of
discernibility against all of $U$. The $\gamma$-bireduct excludes objects whose $B$-indiscernibility
class contains different decisions (e.g., $u_6$ and $u_{14}$ are in the same class as objects with
opposite decisions).
