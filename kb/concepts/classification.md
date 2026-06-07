---
id: concept-classification
type: concept
status: complete
created: 2026-06-04
updated: 2026-06-04
tags: [core, classification]
requires: [concept-decision-table]
see_also: [concept-classification-evaluation, concept-decision-rule]
source: src-thesis-phd
---

# Classification

A classification task is the problem of assigning decision values to objects based on their
attributes; a classification model is the learned function that realizes this assignment.

## Classification Task

In machine learning, a classification task is a type of problem where the goal is to assign a
decision value (or label) to an input object based on its attributes. Depending on the nature of the
labels, the task can be categorized as:

- **Binary classification task**: When there are two mutually exclusive labels from which a choice
  must be made.
- **Multi-class classification task**: When there are three or more labels available.
- **Multi-label classification task**: When the goal is to assign zero or more labels for each input
  object.
- **Probabilistic classification task**: When the goal is to assign a probability distribution over a
  set of decision values, rather than obtaining exact labels.

## Classification Model

A classification model is a type of machine learning model that predicts a decision value (or
decision values) of a given input object. It is an implementation (or realization) of a given
classification task. It can be represented as a function that maps input data objects to predicted
decisions:

$$\hat{d}_{\mathcal{M}} : \mathcal{U} \to \mathcal{D}$$

where $\mathcal{U}$ is a generally understood universe of objects and $\mathcal{D}$ is a set of
available decision values.

In the context of this knowledge base, the input space $\mathcal{U}$ and decision space $\mathcal{D}$
are mapped to the universe of objects $U$ and the value set $V_d$ defined in the
[decision table](../concepts/decision-table.md). Many machine learning methods provide both a
predicted class label and a probability distribution over all possible decision values.

## Remarks

Classical classification models include logistic regression, $k$-nearest neighbors ($k$-NN),
decision trees (ID3, C4.5, CART), and Random Forest. XGBoost represents a state-of-the-art
gradient boosting method widely used for tabular data.

The performance of classification models is assessed using [evaluation
metrics](../concepts/classification-evaluation.md).
