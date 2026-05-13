# Agents Knowledge Base Guidelines

This document provides guidelines for agents maintaining the knowledge base to ensure consistency, machine-readability, and ease of navigation.

## General Rules

- **Language**: All technical definitions, descriptions, and documentation must be written in **English**.
- **Typography**: Avoid special decorative characters or non-standard dashes. Use regular hyphens (`-`) for lists and separators.
- **Formatting**: Use standard Markdown. For mathematical formulas, use LaTeX syntax compatible with KaTeX/MathJax.

## Metadata and Structure

All files in the `knowledgebase/` directory must start with a YAML frontmatter block to provide structured metadata without cluttering the main content.

### Frontmatter Fields
- `tags`: A list of keywords to categorize the content (e.g., `rst`, `ml`, `core`).
- `related`: A list of file paths to other related documents in the knowledge base.

Example:
```markdown
---
tags: [rst, core]
related: [definitions/decision_table.md]
---
# Title
Content...
```

## Symbol Consistency
Always refer to `knowledgebase/notation_and_symbols.md` to ensure consistent use of mathematical symbols.
