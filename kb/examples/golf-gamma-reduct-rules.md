---
id: ex-golf-gamma-reduct-rules
type: example
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [reduction, positive-region, rules, decision-table]
requires:
  [concept-decision-table,
   concept-gamma-decision-reduct,
   concept-positive-region,
   concept-decision-rule]
see_also: [ex-golf-reduct-rules, ex-golf-bireduct-rules]
source: tmp/phd/thesis.tex
---

# Golf Dataset -- Gamma-Decision Reduct Rules

Decision rules generated from $\gamma$-decision reducts for the golf dataset restricted to smaller
attribute subsets. Unlike the full table (which is consistent), restricted subsets make the table
inconsistent, requiring the $\gamma$-decision reduct framework.

## Background: Modified Decision Tables

For an inconsistent table, the $\gamma$-decision reduct construction replaces conflicting decisions
with the special value $\circledast$. Below are the modified tables $\mathbb{A}_B^\gamma$ for the
two restricted subsets.

### $\mathbb{A}_{\{O,T,H\}}^\gamma$

| ID | Outlook  | Temp. | Humidity | $d_{\{O,T,H\}}^\gamma$ |
| -- | -------- | ----- | -------- | :---------------------: |
| 1  | sunny    | hot   | high     | no                      |
| 2  | sunny    | hot   | high     | no                      |
| 3  | overcast | hot   | high     | yes                     |
| 4  | rain     | mild  | high     | $\circledast$           |
| 5  | rain     | cool  | normal   | $\circledast$           |
| 6  | rain     | cool  | normal   | $\circledast$           |
| 7  | overcast | cool  | normal   | yes                     |
| 8  | sunny    | mild  | high     | no                      |
| 9  | sunny    | cool  | normal   | yes                     |
| 10 | rain     | mild  | normal   | yes                     |
| 11 | sunny    | mild  | normal   | yes                     |
| 12 | overcast | mild  | high     | yes                     |
| 13 | overcast | hot   | normal   | yes                     |
| 14 | rain     | mild  | high     | $\circledast$           |

Objects 4, 5, 6, 14 are outside $POS(\{O, T, H\})$ and receive $\circledast$ as decision.

### $\mathbb{A}_{\{T,H,W\}}^\gamma$

| ID | Temp. | Humidity | Wind   | $d_{\{T,H,W\}}^\gamma$ |
| -- | ----- | -------- | ------ | :---------------------: |
| 1  | hot   | high     | weak   | $\circledast$           |
| 2  | hot   | high     | strong | no                      |
| 3  | hot   | high     | weak   | $\circledast$           |
| 4  | mild  | high     | weak   | $\circledast$           |
| 5  | cool  | normal   | weak   | yes                     |
| 6  | cool  | normal   | strong | $\circledast$           |
| 7  | cool  | normal   | strong | $\circledast$           |
| 8  | mild  | high     | weak   | $\circledast$           |
| 9  | cool  | normal   | weak   | yes                     |
| 10 | mild  | normal   | weak   | yes                     |
| 11 | mild  | normal   | strong | yes                     |
| 12 | mild  | high     | strong | $\circledast$           |
| 13 | hot   | normal   | weak   | yes                     |
| 14 | mild  | high     | strong | $\circledast$           |

Objects 1, 3, 4, 6, 7, 8, 12, 14 are outside $POS(\{T, H, W\})$.

## Decision Rules from $\gamma$-Decision Reducts

In both cases the $\gamma$-decision reduct consists of all attributes of the limited table (no
further reduction is possible). All rules have confidence $= 1$ because they are generated from
objects belonging to the positive region.

### Reduct $\{\text{Outlook}, \text{Temperature}, \text{Humidity}\}$

| Rule                                                                                                   |  Support   |
| :----------------------------------------------------------------------------------------------------- | :--------: |
| $(O = \text{overcast}) \land (T = \text{cool}) \land (H = \text{normal}) \Rightarrow (d = \text{yes})$ |  $\{7\}$   |
| $(O = \text{overcast}) \land (T = \text{hot}) \land (H = \text{high}) \Rightarrow (d = \text{yes})$    |  $\{3\}$   |
| $(O = \text{overcast}) \land (T = \text{hot}) \land (H = \text{normal}) \Rightarrow (d = \text{yes})$  |  $\{13\}$  |
| $(O = \text{overcast}) \land (T = \text{mild}) \land (H = \text{high}) \Rightarrow (d = \text{yes})$   |  $\{12\}$  |
| $(O = \text{rain}) \land (T = \text{mild}) \land (H = \text{normal}) \Rightarrow (d = \text{yes})$     |  $\{10\}$  |
| $(O = \text{sunny}) \land (T = \text{cool}) \land (H = \text{normal}) \Rightarrow (d = \text{yes})$    |  $\{9\}$   |
| $(O = \text{sunny}) \land (T = \text{hot}) \land (H = \text{high}) \Rightarrow (d = \text{no})$        | $\{1, 2\}$ |
| $(O = \text{sunny}) \land (T = \text{mild}) \land (H = \text{high}) \Rightarrow (d = \text{no})$       |  $\{8\}$   |
| $(O = \text{sunny}) \land (T = \text{mild}) \land (H = \text{normal}) \Rightarrow (d = \text{yes})$    |  $\{11\}$  |

### Reduct $\{\text{Temperature}, \text{Humidity}, \text{Wind}\}$

| Rule                                                                                                 |  Support   |
| :--------------------------------------------------------------------------------------------------- | :--------: |
| $(T = \text{cool}) \land (H = \text{normal}) \land (W = \text{weak}) \Rightarrow (d = \text{yes})$   | $\{5, 9\}$ |
| $(T = \text{hot}) \land (H = \text{high}) \land (W = \text{strong}) \Rightarrow (d = \text{no})$     |  $\{2\}$   |
| $(T = \text{hot}) \land (H = \text{normal}) \land (W = \text{weak}) \Rightarrow (d = \text{yes})$    |  $\{13\}$  |
| $(T = \text{mild}) \land (H = \text{normal}) \land (W = \text{strong}) \Rightarrow (d = \text{yes})$ |  $\{11\}$  |
| $(T = \text{mild}) \land (H = \text{normal}) \land (W = \text{weak}) \Rightarrow (d = \text{yes})$   |  $\{10\}$  |

## Remarks

Comparing this with the standard [decision reduct rules](golf-reduct-rules.md):

- The full golf table is consistent, so $\gamma$-decision reducts with $B = A$ coincide with
  standard reducts and yield 12 rules each.
- When restricted to subsets like $\{O, T, H\}$ or $\{T, H, W\}$, the table becomes inconsistent
  -- some objects have the same attribute values but different decisions.
- The $\gamma$-decision reduct framework handles this by excluding conflicting objects from
  consideration (they are assigned $\circledast$ in the modified table). Rules are generated only
  from $POS(B)$.
- $\{T, H, W\}$ yields only 5 rules because many objects are outside the positive region (e.g.,
  objects 3 and 4 share values $(hot, high, weak)$ but have decisions "yes" and "yes"... wait,
  actually objects 1 and 3 share $(hot, high, weak)$ but have decisions "no" and "yes" --
  hence both are excluded).
