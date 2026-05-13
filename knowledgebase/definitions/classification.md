---
tags: [ml, core]
related: [definitions/decision_table.md]
---
# Classification

## Classification Task

In machine learning, a classification task is a type of problem where the goal is to assign a decision value (or label) to an input object based on its attributes. Depending on the nature of the labels, the task can be categorized as:

- **Binary classification task**: When there are two mutually exclusive labels from which a choice must be made.
- **Multi-class classification task**: When there are three or more labels available.
- **Multi-label classification task**: When the goal is to assign zero or more labels for each input object.
- **Probabilistic classification task**: When the goal is to assign a probability distribution over a set of decision values, rather than getting the exact labels.

## Classification Model

A classification model is a type of machine learning model that predicts a decision value (or decision values) of a given input object. In other words, it is an implementation (or realization) of a given classification task.

It can be represented as a function that maps input data objects to predicted decisions:
$$\hat{d}_{\text{model}} : \mathcal{U} \rightarrow \mathcal{D}$$

Where:
- $\mathcal{U}$ is the universe of objects.
- $\mathcal{D}$ is the set of available decision values.

***

**Note:** In the context of this project, the general input space $\mathcal{U}$ and decision space $\mathcal{D}$ used in machine learning are mapped to the universe of objects $U$ and the value set of the decision attribute $V_d$ defined in the decision table.
