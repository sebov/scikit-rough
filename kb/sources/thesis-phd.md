---
id: src-thesis-phd
type: source-summary
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [core, bireducts, reducts, rough-sets]
requires: []
see_also:
  [concept-decision-table,
   concept-decision-bireduct,
   concept-gamma-decision-bireduct,
   concept-epsilon-decision-bireduct,
   concept-bireduct-ensemble,
   concept-temporal-bireduct]
source: "tmp/phd/thesis.tex (PhD dissertation, 6572 lines + 30 include files)"
---

# PhD Thesis: Decision Bireducts in Rough Set Theory

Primary source for the knowledge base. A doctoral dissertation covering the theory of decision
bireducts, their variants (gamma, epsilon), computational complexity, and algorithms.

## Source Structure

- **Chapter 1**: Preliminaries -- classification, evaluation metrics.
- **Chapter 2**: Foundations of decision reducts -- indiscernibility, approximations, consistency,
  formulae, decision rules, reducts (classical, gamma, approximate), NP-hardness foundations.
- **Chapter 3**: Decision bireducts -- definition, monotonicity, Boolean formula characterisation,
  diagonal table transformation, gamma-bireducts, epsilon-bireducts, ensembles, NP-hardness
  results.
- **Chapter 4**: Algorithms -- ordering, sampling, temporal bireducts for data streams.
- **Chapters 5-7**: Case study, feature importance, conclusions (no new formal content -- not
  ingested).
- **Include files** (`tmp/phd/include/`): 30 LaTeX files with detailed tables, formulae, and
  example computations. All examples extracted to `kb/examples/`.

## Bibliography

Reference file: `tmp/phd/thesisbib.bib`. All citation titles verified against this file during
ingestion.

## Key Notation

47 symbols registered in `kb/notation.md` from the thesis preamble (lines 150-359). Custom LaTeX
commands decoded in the notation file's Source-to-KB Translation Notes section.

## Ingestion Summary

- **Concepts**: 21 files in `kb/concepts/`.
- **Propositions**: 30 files in `kb/propositions/`.
- **Examples**: 13 files in `kb/examples/`.
- **Total wiki pages from this source**: 64 (excluding this source-summary).
- **Status**: COMPLETE (2026-06-07).
