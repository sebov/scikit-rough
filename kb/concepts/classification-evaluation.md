---
id: concept-classification-evaluation
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, evaluation, classification]
requires: [concept-classification, concept-decision-table]
see_also: [concept-decision-rule]
source: src-thesis-phd
---

# Classification Evaluation

Evaluation metrics quantify the performance of a classification model on a given decision table.
This file covers the general evaluation framework and standard metrics used for binary and
multi-class classification.

## Evaluation Metric

An evaluation metric $eval : \mathcal{M} \times \mathbb{A} \to \mathbb{R}$ is a function that
quantifies the performance of a machine learning model $\mathcal{M}$ on a given decision table
$\mathbb{A} = (U, A \cup \{d\})$. The application of the evaluation metric is constrained to input
arguments that are compatible: the model must be able to recognize the representation of objects in
$\mathbb{A}$ and make predictions on them.

## Model Outcomes (Confusion Matrix)

Let a binary classification task be defined on $\mathbb{A}$ and let a classification model
$\mathcal{M}$ be given. Choose a distinguished decision value $v_{d_{relevant}} \in V_d$ to indicate
"relevant objects". The set of objects $U$ is partitioned into four disjoint subsets based on actual
and predicted decisions:

$$
\begin{aligned}
TP_{\mathcal{M}, \mathbb{A}} &= \{u \in U : d(u) = v_{d_{relevant}} \land d(u) = \hat{d}_{\mathcal{M}}(u)\} \\
FN_{\mathcal{M}, \mathbb{A}} &= \{u \in U : d(u) = v_{d_{relevant}} \land d(u) \neq \hat{d}_{\mathcal{M}}(u)\} \\
TN_{\mathcal{M}, \mathbb{A}} &= \{u \in U : d(u) \neq v_{d_{relevant}} \land d(u) = \hat{d}_{\mathcal{M}}(u)\} \\
FP_{\mathcal{M}, \mathbb{A}} &= \{u \in U : d(u) \neq v_{d_{relevant}} \land d(u) \neq \hat{d}_{\mathcal{M}}(u)\}
\end{aligned}
$$

When $\mathcal{M}$ and $\mathbb{A}$ are clear from context, the notation $TP, FN, TN, FP$ is used.

## Precision

Precision is the fraction of objects classified as relevant that are actually relevant:

$$
precision(\mathcal{M}, \mathbb{A}) =
\begin{cases}
\frac{\lvert TP \rvert}{\lvert TP \rvert + \lvert FP \rvert} & \text{if } \lvert TP \rvert + \lvert FP \rvert > 0 \\
0 & \text{otherwise}
\end{cases}
$$

## Recall (Sensitivity, True Positive Rate)

Recall (also called sensitivity or $TPR$) is the fraction of relevant objects correctly classified:

$$
recall(\mathcal{M}, \mathbb{A}) = sensitivity(\mathcal{M}, \mathbb{A}) = TPR(\mathcal{M}, \mathbb{A}) =
\begin{cases}
\frac{\lvert TP \rvert}{\lvert TP \rvert + \lvert FN \rvert} & \text{if } \lvert TP \rvert + \lvert FN \rvert > 0 \\
0 & \text{otherwise}
\end{cases}
$$

## F1 Score

F1 score is the harmonic mean of precision and recall:

$$
\operatorname{F1-score}(\mathcal{M}, \mathbb{A}) =
\begin{cases}
2 \cdot \frac{precision(\mathcal{M}, \mathbb{A}) \cdot recall(\mathcal{M}, \mathbb{A})}
                {precision(\mathcal{M}, \mathbb{A}) + recall(\mathcal{M}, \mathbb{A})}
    & \text{if } precision(\mathcal{M}, \mathbb{A}) + recall(\mathcal{M}, \mathbb{A}) > 0 \\
0 & \text{otherwise}
\end{cases}
$$

## Specificity (True Negative Rate)

Specificity (or $TNR$) is the fraction of non-relevant objects correctly classified. It can be
viewed as recall with the distinguished decision value swapped:

$$
specificity(\mathcal{M}, \mathbb{A}) = TNR(\mathcal{M}, \mathbb{A}) =
\begin{cases}
\frac{\lvert TN \rvert}{\lvert TN \rvert + \lvert FP \rvert} & \text{if } \lvert TN \rvert + \lvert FP \rvert > 0 \\
0 & \text{otherwise}
\end{cases}
$$

## Class-Specific Recall

Class-specific recall generalizes recall and specificity to multi-class tasks. For a given decision
value $v_d \in V_d$:

$$
recall_{v_d}(\mathcal{M}, \mathbb{A}) =
\begin{cases}
\frac
    {\lvert\{u \in U : d(u) = v_d \land d(u) = \hat{d}_{\mathcal{M}}(u)\}\rvert}
    {\lvert\{u \in U : d(u) = v_d\}\rvert}
    & \text{if } \lvert\{u \in U : d(u) = v_d\}\rvert > 0 \\
0 & \text{otherwise}
\end{cases}
$$

## Accuracy

Accuracy is the fraction of objects classified correctly. For binary classification using confusion
matrix notation:

$$
ACC(\mathcal{M}, \mathbb{A}) = \frac{\lvert TP \rvert + \lvert TN \rvert}
                                   {\lvert TP \rvert + \lvert FN \rvert + \lvert TN \rvert + \lvert FP \rvert}
$$

In the general multi-class form:

$$
ACC(\mathcal{M}, \mathbb{A}) = \frac
    {\lvert\{u \in U : d(u) = \hat{d}_{\mathcal{M}}(u)\}\rvert}
    {\lvert U \rvert}
$$

## Balanced Accuracy

Accuracy can be misleading on imbalanced data sets where decision classes have unequal sizes. A model
always predicting the majority class may achieve high accuracy but fail to capture minority classes.

Balanced accuracy addresses this by averaging class-specific recall values. For binary
classification:

$$
BAC(\mathcal{M}, \mathbb{A}) = \frac{TPR(\mathcal{M}, \mathbb{A}) + TNR(\mathcal{M}, \mathbb{A})}{2}
$$

In the general multi-class form:

$$
BAC(\mathcal{M}, \mathbb{A}) = \frac{\sum_{v_d \in V_d} recall_{v_d}(\mathcal{M}, \mathbb{A})}
                                    {\lvert V_d \rvert}
$$

## Remarks

For probabilistic classification models, the predicted probabilities must be converted to class
labels via a threshold. The ROC (Receiver Operating Characteristic) curve visualizes performance
across all threshold values, and AUC (Area Under the ROC Curve) provides a threshold-independent
scalar measure of class separability.

Cross-validation (e.g., $k$-fold, leave-one-out, leave-one-group-out) is the standard technique for
estimating model performance by repeatedly splitting data into training and test subsets.
