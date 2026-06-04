---
tags: [ml, core, evaluation]
related: [definitions/classification.md, definitions/decision_table.md, definitions/decision_rules.md]
---
# Evaluation

## Evaluation Metric

An evaluation metric is a function that quantifies the performance of a machine learning model on a
given data table: $$eval : \text{model} \times \text{data table} \rightarrow \mathbb{R}$$

The application of the evaluation metric is constrained to input arguments that are compatible with
each other. Specifically, the model must be able to recognize the representation of objects in the
data table and make predictions on them.
