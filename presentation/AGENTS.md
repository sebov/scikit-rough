# Agent Instructions -- Presentation

## Title

Algorithms for Approximate Reducts: Iterative Attribute Selection

## Overview

This directory contains a presentation prepared for a workshop. The focus is on
practical aspects rather than theory, though some theoretical background is
included for context.

## Format

The presentation is authored as a Jupyter Notebook (Python) and rendered via
**Quarto** in **revealjs** mode. A PDF backup format is also supported.

## Content

The presentation covers implementation approaches for searching approximate
reducts (not bireducts), with a focus on the greedy strategy as implemented in
the `skrough` package (`src/`). Practical aspects discussed include:

- The modular skeleton of the computation (stop conditions, evaluation
  functions, etc.)
- Performance profiling -- what takes the most time (e.g., `group_index`
  representation of indiscernibility equivalence classes, and how to
  iteratively add new attributes without rebuilding the index from scratch)
- Comparison of different implementations and their runtimes

## Sources

The theoretical foundation follows the definitions and propositions in
`knowledgebase/`.

## Style

All code comments, docstrings, and text in this directory must be written in
**English**. Use regular hyphens (`-`) instead of em-dashes or en-dashes. Avoid
emojis and decorative symbols.

Docstrings follow the Google style format.

## Checkpoint

Last session used `grill-with-docs` skill to establish presentation structure and
domain glossary. Key files to resume from:

- `structure.md` -- full slide outline, resolved decisions, open questions
- `CONTEXT.md` -- glossary of domain terms
- `presentation.ipynb` -- the notebook being built (to be executed via `uv run jupyter nbconvert --to notebook --execute --inplace presentation.ipynb`, then rendered via `quarto render`)

To resume work: say "wznawiamy" or "kontynuujemy od checkpointu".

### Build Commands

```bash
# Execute notebook (from presentation/ dir)
uv run jupyter nbconvert --to notebook --execute --inplace presentation.ipynb

# Render to revealjs
quarto render presentation.ipynb --to revealjs --output-dir _output --no-execute
```
