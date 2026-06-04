---
id: ex-golf-reduct-rules
type: example
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [reduction, rules, decision-table]
requires: [concept-decision-table, concept-decision-reduct, concept-decision-rule]
see_also: [ex-golf-bireduct-rules]
source: tmp/phd/thesis.tex
---

# Golf Dataset -- Decision Reduct Rules

Decision rules generated from the two decision reducts of the golf dataset
(see [Decision Table](../concepts/decision-table.md)). All rules have confidence = 1 because the
golf dataset is consistent.

## Reduct $\{\text{Outlook}, \text{Temperature}, \text{Wind}\}$

| Rule                                                                                                   |   Support   |
| :----------------------------------------------------------------------------------------------------- | :---------: |
| $(O = \text{overcast}) \land (T = \text{cool}) \land (W = \text{strong}) \Rightarrow (d = \text{yes})$ |   $\{7\}$   |
| $(O = \text{overcast}) \land (T = \text{hot}) \land (W = \text{weak}) \Rightarrow (d = \text{yes})$    | $\{3, 13\}$ |
| $(O = \text{overcast}) \land (T = \text{mild}) \land (W = \text{strong}) \Rightarrow (d = \text{yes})$ |  $\{12\}$   |
| $(O = \text{rain}) \land (T = \text{cool}) \land (W = \text{strong}) \Rightarrow (d = \text{no})$      |   $\{6\}$   |
| $(O = \text{rain}) \land (T = \text{cool}) \land (W = \text{weak}) \Rightarrow (d = \text{yes})$       |   $\{5\}$   |
| $(O = \text{rain}) \land (T = \text{mild}) \land (W = \text{strong}) \Rightarrow (d = \text{no})$      |  $\{14\}$   |
| $(O = \text{rain}) \land (T = \text{mild}) \land (W = \text{weak}) \Rightarrow (d = \text{yes})$       | $\{4, 10\}$ |
| $(O = \text{sunny}) \land (T = \text{cool}) \land (W = \text{weak}) \Rightarrow (d = \text{yes})$      |   $\{9\}$   |
| $(O = \text{sunny}) \land (T = \text{hot}) \land (W = \text{strong}) \Rightarrow (d = \text{no})$      |   $\{2\}$   |
| $(O = \text{sunny}) \land (T = \text{hot}) \land (W = \text{weak}) \Rightarrow (d = \text{no})$        |   $\{1\}$   |
| $(O = \text{sunny}) \land (T = \text{mild}) \land (W = \text{strong}) \Rightarrow (d = \text{yes})$    |  $\{11\}$   |
| $(O = \text{sunny}) \land (T = \text{mild}) \land (W = \text{weak}) \Rightarrow (d = \text{no})$       |   $\{8\}$   |

## Reduct $\{\text{Outlook}, \text{Humidity}, \text{Wind}\}$

| Rule                                                                                                     |   Support   |
| :------------------------------------------------------------------------------------------------------- | :---------: |
| $(O = \text{overcast}) \land (H = \text{high}) \land (W = \text{strong}) \Rightarrow (d = \text{yes})$   |  $\{12\}$   |
| $(O = \text{overcast}) \land (H = \text{high}) \land (W = \text{weak}) \Rightarrow (d = \text{yes})$     |   $\{3\}$   |
| $(O = \text{overcast}) \land (H = \text{normal}) \land (W = \text{strong}) \Rightarrow (d = \text{yes})$ |   $\{7\}$   |
| $(O = \text{overcast}) \land (H = \text{normal}) \land (W = \text{weak}) \Rightarrow (d = \text{yes})$   |  $\{13\}$   |
| $(O = \text{rain}) \land (H = \text{high}) \land (W = \text{strong}) \Rightarrow (d = \text{no})$        |  $\{14\}$   |
| $(O = \text{rain}) \land (H = \text{high}) \land (W = \text{weak}) \Rightarrow (d = \text{yes})$         |   $\{4\}$   |
| $(O = \text{rain}) \land (H = \text{normal}) \land (W = \text{strong}) \Rightarrow (d = \text{no})$      |   $\{6\}$   |
| $(O = \text{rain}) \land (H = \text{normal}) \land (W = \text{weak}) \Rightarrow (d = \text{yes})$       | $\{5, 10\}$ |
| $(O = \text{sunny}) \land (H = \text{high}) \land (W = \text{strong}) \Rightarrow (d = \text{no})$       |   $\{2\}$   |
| $(O = \text{sunny}) \land (H = \text{high}) \land (W = \text{weak}) \Rightarrow (d = \text{no})$         | $\{1, 8\}$  |
| $(O = \text{sunny}) \land (H = \text{normal}) \land (W = \text{strong}) \Rightarrow (d = \text{yes})$    |  $\{11\}$   |
| $(O = \text{sunny}) \land (H = \text{normal}) \land (W = \text{weak}) \Rightarrow (d = \text{yes})$      |   $\{9\}$   |

## Remarks

Each reduct yields 12 deterministic rules (confidence = 1). Together they cover all 14 objects of the
golf dataset. A decision reduct $B \subseteq A$ guarantees that $U$ can be partitioned by rules whose
predecessors use only attributes from $B$.
