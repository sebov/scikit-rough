---
tags: [index, guidelines]
related: [index.md, notation_and_symbols.md]
---
# Agents Knowledge Base Guidelines

This document provides guidelines for agents maintaining the knowledge base to ensure consistency,
machine-readability, and ease of navigation.

## General Rules

- **Language**: All technical definitions, descriptions, and documentation must be written in
  **English**.
- **Typography**: Avoid special decorative characters or non-standard dashes. Use regular hyphens
  (`-`) for lists and separators.
- **Formatting**: Use standard Markdown. For mathematical content, use LaTeX syntax compatible with
  KaTeX/MathJax. This applies to all mathematical notation, including inline symbols (e.g., `$U$`,
  `$a \in A$`), not only block equations. Do not use backticks or Unicode characters as substitutes
  for math symbols.

### Formatting Nice-to-Haves

The following are soft guidelines. Apply them when convenient, but do not force them at the expense
of readability:

- **Line length**: Aim for approximately 100 characters per line when wrapping prose.
- **Blank lines around lists**: Place a blank line before a list and after the preceding paragraph.
- **Blank lines after headings**: Place a blank line between a heading (`## Title`) and the content
  that follows it.

## Metadata and Structure

All files in the `knowledgebase_old/` directory must start with a YAML frontmatter block to provide
structured metadata without cluttering the main content.

### Frontmatter Fields

- `tags`: A list of keywords to categorize the content (e.g., `rst`, `ml`, `core`, `reduction`,
  `bireducts`, `proposition`, `complexity`, `evaluation`).
- `related`: A list of file paths to both forward and backward references. Includes all files that
  are substantively connected.

Example:

```markdown
---
tags: [rst, core]
related: [definitions/decision_table.md]
---
# Title
Content...
```

## Naming Conventions

- **Definition files**: Use short, prefix-free names matching the concept (e.g., `reducts.md`,
  `approximate_reducts.md`, `bireducts.md`). Avoid redundant prefixes like `decision_` when the
  domain is clear.
- **Proposition files**: Place in `propositions/` with descriptive names (e.g.,
  `gamma_decision_reduct_characterization.md`, `approximate_reduct_complexity.md`).

## Propositions

- **Inline** in definition files: Short propositions that directly support a nearby definition
  (e.g., monotonicity properties, equivalence proofs of a few lines).
- **Separate files** in `propositions/`: Standalone results with substantial proofs, or results
  referenced from multiple definition files.

## Tables

- Use standard Markdown table syntax with column alignment via dashes and colons.
- For tabular data with uniform values, center-align columns (`:---:`).
- For mixed content (text + math), align logically (e.g., left for text, center for values).

## Symbol Consistency

Always refer to `knowledgebase/notation_and_symbols.md` to ensure consistent use of mathematical
symbols.
