# Knowledge Base Architecture and Agent Protocol

This document is the **schema layer** for the rough set theory knowledge base. It defines the
structure, conventions, and workflows that all executor agents must follow when ingesting sources,
answering queries, or maintaining the wiki.

The knowledge base is a modular, distributed collection of small, interconnected Markdown files
covering rough set theory (RST). It is designed for two audiences:

1. **LLM agents** (primary): producing documentation, generating educational material, defining
   concepts, storing theorems, constructing proofs, and verifying correctness.
2. **Human readers** (secondary): browsing via plain Markdown without specialized tools.

---

## Executor Agent Role

You are a knowledge base executor agent. Your role is to ingest raw source material and produce
well-structured, atomic Markdown files that conform to the schema defined in this document.

**Read this document before every operation.** It is the canonical rules document. If any
instruction from the operator conflicts with rules defined here, these rules take precedence.

### Your Responsibilities

1. **Extract**: identify all concepts, propositions, examples, and notation in the source
   material.
2. **Translate**: convert the source's notation to match the conventions in `kb/notation.md`. If
   the source uses a symbol already defined differently in `notation.md`, rewrite using the KB
   convention. If the symbol is genuinely new, add it to `notation.md`.
3. **Create or Update**: for each identified concept:
   - If it already exists in the KB: update the existing file with new information, following
     the incremental update strategy (Section 11).
   - If it does not exist: create a new file using the template in `kb/template.md`.
4. **Link**: populate `requires` (direct prerequisites, 2-5 entries) and `see_also` (related
   concepts, 3-8 entries) in frontmatter. Use in-body Markdown links freely.
5. **Verify**: run through the quality checklist (Section 12) before finalizing.
6. **Log**: update `kb/index.md` with new/modified entries. Append an entry to `kb/log.md`.

### Output Format

For each file you create or modify, output the complete file content including frontmatter.
Clearly label each file with its path (e.g., `kb/concepts/indiscernibility.md`).

After all files are produced, output:
1. Updated `notation.md` entries (if any new symbols).
2. Updated `index.md` entries (new/modified files with id, link, one-line summary).
3. Log entry for `kb/log.md`.

---

## 1. Directory Structure

The knowledge base uses a shallow, category-based directory layout. Organization is driven by
metadata (the `type` field in frontmatter), not by deep folder hierarchies.

```
kb/
  AGENTS.md              -- This file. The schema layer and executor instructions. Read before any operation.
  index.md               -- Content-oriented catalog of all wiki pages (updated on every ingest).
  log.md                 -- Append-only chronological journal of all operations.
  notation.md            -- Centralized registry of mathematical symbols and notation conventions.
  ingestion.md           -- Per-source ingestion tracking: progress, decisions, pending items.
  template.md            -- File template with correct frontmatter and heading structure.
  concepts/              -- Core definitions, concepts, and foundational material.
  propositions/          -- Theorems, lemmas, propositions with substantial proofs.
  examples/              -- Worked examples, counterexamples, illustrative datasets.
  sources/               -- Source-summary files: provenance metadata + distilled key insights.
  queries/               -- Filed query results that compound over time.
```

### Organization Principles

- **Shallow hierarchy**: at most one level of subdirectories. No nested folders within
  `concepts/`, `propositions/`, etc.
- **Metadata-driven categorization**: the `type` field in frontmatter determines what a file is,
  not its directory. A file in `concepts/` could theoretically have `type: proposition` if it was
  misplaced -- agents should fix this during lint.
- **Self-containment**: the `kb/` directory is fully self-contained. It can be extracted into a
  separate repository and function independently. References to external code (e.g., `src/skrough/`)
  are allowed only as annotations or external links, never as structural dependencies.
- **Git-native**: the entire knowledge base is a git repository of Markdown files. One logical
  change per commit. Commit messages should be descriptive (e.g., "ingest: Pawlak 1982, ch. 2" or
  "update: add counterexample to concept-indiscernibility").

---

## 2. File Template

All wiki pages must follow this template. A copy is available at `kb/template.md`.

```markdown
---
id: <unique-identifier>
type: <concept | proposition | example | source-summary | query-result>
status: <draft | complete | reviewed | deprecated>
created: <YYYY-MM-DD>
updated: <YYYY-MM-DD>
tags: [<tag1>, <tag2>, ...]
requires: [<id-of-prerequisite-1>, <id-of-prerequisite-2>, ...]
see_also: [<id-of-related-1>, <id-of-related-2>, ...]
source: <source-summary-id or citation>
---

# <Title>

<One-line summary of the concept, proposition, or example.>

## Definition

<The formal definition, using LaTeX math notation.>

## Intuition

<Plain-language explanation of what the concept means and why it matters.>

## Example

<A concrete example illustrating the concept.>

## Counterexample

<If applicable, a case where the concept does NOT apply or a common misconception.>

## Theorem

<If applicable, a theorem statement.>

## Proof

<If applicable, the proof.>

## Remarks

<Additional notes, connections to other concepts, historical context, or open questions.>
```

### Heading Structure

- **H1** (`#`): exactly one per file -- the title. Must match the `id` field semantically.
- **H2** (`##`): major sections (Definition, Intuition, Example, Theorem, Proof, Remarks).
- **H3** (`###`): subsections within major sections (e.g., "Alternative Formulation", "Special
  Case").
- **H4** (`####`): rarely needed. Use only for deeply nested sub-cases.

Not all sections are required. A concept file may omit Counterexample, Theorem, and Proof. A
proposition file may omit Intuition and Example if the result is purely technical. Use judgment.

---

## 3. Naming Conventions

### File Names

- **Lowercase with hyphens**: `indiscernibility.md`, `decision-table.md`,
  `gamma-decision-reduct.md`.
- **No prefixes**: avoid `concept-indiscernibility.md` -- the directory already indicates the
  category. The `id` field in frontmatter carries the unique identifier.
- **Descriptive proposition names**: `reducts-bireducts-link.md`,
  `gamma-reduct-boolean-formula.md`.
- **Source summaries**: use author-year format: `pawlak-1982.md`, `skowron-2001.md`.
- **Query results**: use a descriptive slug: `reduct-vs-approximate-reduct-comparison.md`.
- **No special characters**: only lowercase letters, digits, and hyphens. No spaces, underscores,
  or Unicode.

### Identifier (`id` field)

- **Format**: `<type-prefix>-<slug>`, e.g., `concept-indiscernibility`,
  `prop-gamma-reduct-characterization`, `ex-golf-dataset`, `src-pawlak-1982`.
- **Type prefixes**: `concept-`, `prop-`, `ex-`, `src-`, `query-`.
- **Slug**: matches the file name (without extension), e.g., file `gamma-decision-reduct.md` has
  id `concept-gamma-decision-reduct`.
- **Uniqueness**: every `id` must be globally unique across the entire knowledge base. The
  executor agent must check for collisions before creating a new file.

---

## 4. Metadata Schema

All frontmatter fields and their semantics:

| Field       | Required | Type              | Description                                                                 |
| :---------- | :------: | :---------------- | :-------------------------------------------------------------------------- |
| `id`        |   Yes    | string            | Globally unique identifier. Format: `<type-prefix>-<slug>`.                |
| `type`      |   Yes    | enum              | One of: `concept`, `proposition`, `example`, `source-summary`, `query-result`. |
| `status`    |   Yes    | enum              | One of: `draft`, `complete`, `reviewed`, `deprecated`.                     |
| `created`   |   Yes    | date (YYYY-MM-DD) | Date the file was first created.                                           |
| `updated`   |   Yes    | date (YYYY-MM-DD) | Date of the most recent modification.                                      |
| `tags`      |   Yes    | list of strings   | Categorization keywords (e.g., `core`, `reduction`, `evaluation`, `rules`). |
| `requires`  |    No    | list of ids       | Prerequisite concepts the reader must understand first.                    |
| `see_also`  |    No    | list of ids       | Related concepts for further reading (horizontal links).                   |
| `source`    |    No    | string            | Source-summary `id` (wiki pages) or citation/path (source-summaries). |

### Field Semantics

- **`status` lifecycle**: `draft` (initial creation, may be incomplete) -> `complete` (all
  sections filled, notation verified) -> `reviewed` (verified by lint or human review) ->
  `deprecated` (superseded by newer content, kept for historical reference with a note).
- **`requires`**: encodes a dependency ordering. If concept B depends on concept A, then B's
  `requires` field lists A's `id`. This enables topological sorting for reading order. Keep this
  list short -- only direct prerequisites, not transitive ones.
- **`see_also`**: horizontal cross-references to related but non-prerequisite concepts. This
  replaces the old `related` field that listed every file. Keep this list focused -- at most 5-8
  entries. If a concept is related to many others, it should be linked from `index.md` instead.
- **`tags`**: use a controlled vocabulary. Core tags: `core`, `reduction`, `approximation`,
  `rules`, `evaluation`, `consistency`, `complexity`, `bireducts`, `positive-region`,
  `indiscernibility`, `decision-table`. Add new tags only when no existing tag fits.
- **`source`**: for wiki pages (concepts, propositions, examples), this field holds the `id` of
  the source-summary file in `kb/sources/` that documents the origin of the content (e.g.,
  `src-thesis-phd`). For source-summary files themselves, this field holds the original external
  file path or bibliographic citation. This convention ensures self-containment: the KB never
  depends on external file paths that may disappear.

---

## 5. Linking Rules

### Internal Links

- **In-body links**: use standard Markdown links with relative paths:
  `[indiscernibility](../concepts/indiscernibility.md)`.
- **Frontmatter links**: use `id` values in `requires` and `see_also` fields. Agents resolve
  these to file paths by scanning the knowledge base.
- **No bidirectional enforcement**: do NOT maintain mirror links. If file A links to file B in
  its `see_also`, file B does NOT need to link back to A. The `index.md` file serves as the
  central hub for discoverability.

### Scalability Strategy

The old knowledge base listed every file in the `related` field of `index.md`. This does not
scale. The new approach:

- **`index.md`** is a curated, categorized catalog. Each page is listed with its `id`, a link, and
  a one-line summary. Organized by category (concepts, propositions, examples, sources).
- **`requires`** encodes vertical (prerequisite) dependencies. Kept short (2-5 entries).
- **`see_also`** encodes horizontal (related) links. Kept focused (3-8 entries).
- **In-body links** are used freely within the prose to reference other concepts. These are the
  primary navigation mechanism for human readers.

### Avoiding Dead Ends

- Every file must have at least one inbound link (from `index.md` at minimum).
- Every file must have at least one outbound link (in `requires`, `see_also`, or in-body).
- Orphan detection is part of the Lint operation (see Section 10).

---

## 6. Atomicity Criteria

### What is One Concept?

A single file should cover **one primary mathematical concept** -- typically one definition and
its immediate supporting material (intuition, example, basic properties).

### When to Create a Separate File

Create a separate file when:

- A proposition has a **substantial proof** (more than 20 lines of reasoning).
- A result is **referenced from multiple definition files** (avoids duplication).
- An example is **complex enough** to warrant its own file (e.g., a full worked dataset like the
  golf example).
- A concept has **multiple distinct formulations** that each deserve detailed treatment.

### When to Inline

Inline a proposition or remark within a concept file when:

- It is **short** (under 20 lines including proof).
- It **directly supports** the definition (e.g., a monotonicity property, a simple equivalence).
- It would be **orphaned** as a standalone file (no other file would reference it).

### File Size Threshold

- **Target**: 100-200 lines per file.
- **Hard limit**: if a file approaches 250 lines, split it. Move substantial propositions to
  `propositions/`, move examples to `examples/`, or split a large concept into sub-concepts.
- **Splitting procedure**: when splitting, create new files with their own `id`, update the
  original file's `see_also` to reference the new files, update `index.md`, and log the split in
  `log.md`.

---

## 7. Proposition Placement

### Decision Rule

```
Is the proposition short (< 20 lines) AND only relevant to this one concept?
  -> Inline in the concept file, under a ## Proposition or ## Theorem heading.

Is the proposition substantial (>= 20 lines) OR referenced by multiple concepts?
  -> Separate file in propositions/.
```

### Proposition File Structure

Proposition files use the same template but emphasize:

- **Background**: context and motivation for the result.
- **Statement**: the formal proposition/theorem statement.
- **Proof**: the complete proof.
- **Consequences**: corollaries or implications (if any).

---

## 8. Notation Management

### Centralized Notation File

The file `kb/notation.md` is the single source of truth for all mathematical symbols used across
the knowledge base. It is structured as a table:

```markdown
| Symbol | Name | Description | First Used In |
| :----- | :--- | :---------- | :------------ |
| $U$ | Universe | The universe of objects | concept-decision-table |
| $A$ | Conditional attributes | Set of conditional attributes | concept-decision-table |
| $d$ | Decision attribute | The distinguished decision attribute | concept-decision-table |
| $\mathbb{A}$ | Decision table | A decision table $(U, A \cup \{d\})$ | concept-decision-table |
```

### Notation Consistency Protocol

1. **Before creating a new file**: check `notation.md` for existing symbol definitions.
2. **When ingesting a source**: translate the source's notation to match the knowledge base
   conventions. If the source uses $X$ where the KB uses $U$, rewrite as $U$.
3. **When introducing a new symbol**: add it to `notation.md` with its name, description, and
   the `id` of the file where it first appears.
4. **When a source uses genuinely new notation** (not a rename of an existing symbol): add it to
   `notation.md` and use it consistently in all new files.
5. **Conflict**: if two sources use different symbols for the same concept, the first-ingested
   convention wins. The second source's notation is noted in `notation.md` as an alternative but
   not used in wiki content. Log the conflict in `log.md`.

### Notation Rules

- All mathematical notation uses LaTeX syntax (`$...$` for inline, `$$...$$` for block).
- Never use backticks or Unicode characters as substitutes for math symbols.
- Assume output will be rendered by KaTeX or MathJax.

---

## 9. Conflict Resolution Protocol

When ingesting a new source, the executor agent may encounter claims that contradict existing
wiki content (e.g., a different definition, a conflicting theorem, updated terminology).

### Detection

The agent detects contradictions by:

1. Comparing new definitions against existing definitions for the same concept.
2. Checking if new theorems produce different results from existing theorems under the same
   conditions.
3. Noting terminology differences (same concept, different name).

### Resolution Procedure

1. **Do NOT silently replace** existing content.
2. **Flag-and-append**: add the new formulation to the existing file under a clearly marked
   subsection:
   - `### Alternative Formulation` (for a different but valid definition).
   - `### Contradicting Claim` (for a genuine contradiction).
3. **Annotate**: include the source citation, the date of ingestion, and a brief explanation of
   the discrepancy.
4. **Log**: add an entry to `log.md` with type `conflict`:
   ```
   ## [YYYY-MM-DD] conflict | <concept-id>
   New source <source-id> contradicts existing definition of <concept>.
   Existing: <brief description>. New: <brief description>.
   Resolution: flagged for human review.
   ```
5. **Status**: set the file's `status` to `draft` if the contradiction is unresolved, signaling
   that human review is needed.

### Terminology Conflicts

If a source uses a different name for an existing concept:

- Keep the existing name as the primary name.
- Add the alternative name in the Remarks section: "Also known as <alternative name> in
  <source>."
- Do NOT create a duplicate file.

---

## 10. Operations

### Ingest

**Trigger**: a new raw source is added to the input collection (LaTeX, PDF, Markdown, or code
fragment).

**Procedure**:

1. Read the source material thoroughly.
2. Identify all concepts, propositions, examples, and notation present in the source.
3. For each identified concept:
   - Check if it already exists in the knowledge base (search `index.md` and scan `concepts/`).
   - If it exists: update the existing file with new information, cross-references, or
     alternative formulations. Follow the conflict resolution protocol if contradictions arise.
   - If it does not exist: create a new file using the template. Assign a unique `id`, set
     `status: draft`, populate all applicable sections.
4. Update `notation.md` with any new symbols.
5. Update `index.md` with new or modified entries.
6. Append an entry to `log.md`:
   ```
   ## [YYYY-MM-DD] ingest | <source-title-or-identifier>
   Created: <list of new file ids>.
   Updated: <list of modified file ids>.
   New symbols: <list of new notation entries>.
   Conflicts: <list of flagged contradictions, if any>.
   ```
7. Set all new files to `status: complete` after verification (see Quality Checklist, Section
   12).

### Query

**Trigger**: a question is asked against the knowledge base.

**Procedure**:

1. Read `index.md` to identify relevant pages.
2. Read the relevant pages.
3. Synthesize an answer with citations to specific file `id`s and section headings.
4. If the answer is substantial or reusable, file it as a new page in `queries/` with
   `type: query-result`.
5. Append an entry to `log.md`:
   ```
   ## [YYYY-MM-DD] query | <brief-question-summary>
   Pages consulted: <list of file ids>.
   Result filed as: <query-result-id or "not filed">.
   ```

### Lint

**Trigger**: periodic health check (requested by human operator or scheduled).

**Checks**:

- **Orphan pages**: files with no inbound links (not listed in `index.md` or any other file's
  `requires`/`see_also`).
- **Dead links**: references to non-existent `id`s in `requires`, `see_also`, or in-body links.
- **Stale content**: files with `status: draft` older than 30 days.
- **Notation drift**: symbols used in files but not registered in `notation.md`.
- **Missing cross-references**: concepts that reference each other in-body but not in
  `see_also`.
- **Contradictions**: flagged conflicts that remain unresolved.
- **Oversized files**: files exceeding 250 lines.
- **Missing sections**: concept files without a Definition section, proposition files without a
  Statement section.
- **Frontmatter completeness**: all required fields present and valid.

**Procedure**:

1. Run all checks.
2. Produce a lint report (can be output to terminal or filed as a query result).
3. Append an entry to `log.md`:
   ```
   ## [YYYY-MM-DD] lint | Health check
   Orphans: <count>. Dead links: <count>. Stale drafts: <count>.
   Notation drift: <count>. Unresolved conflicts: <count>.
   Oversized files: <count>. Missing sections: <count>.
   ```

---

## 11. Incremental Update Strategy

### Adding Content to Existing Files

1. **Read the existing file** thoroughly before making changes.
2. **Preserve existing content**: do not delete or rewrite established definitions unless
   following the conflict resolution protocol.
3. **Insert new material** in the appropriate section. If adding a new subsection, place it
   after existing subsections of the same level.
4. **Update `updated` date** in frontmatter.
5. **Update `see_also`** if new cross-references emerge.
6. **Update `index.md`** if the file's one-line summary changes.

### Splitting Oversized Files

1. Identify the natural split point (e.g., a substantial proposition, a large example).
2. Create new file(s) with unique `id`s.
3. Move the relevant content to the new file(s).
4. In the original file, replace the moved content with a brief summary and a link to the new
   file.
5. Update `requires` and `see_also` in both files.
6. Update `index.md`.
7. Log the split in `log.md`.

### Backlink Updates

- When file A is created and references file B (in `requires` or `see_also`), do NOT add a
  reciprocal link in file B.
- File B will discover file A through `index.md` or through lint operations that detect missing
  cross-references.
- This prevents the scalability problem of the old knowledge base where every file listed every
  other file.

---

## 12. Quality Checklist

The executor agent must verify the following before finalizing any output:

### Per-File Checks

- [ ] Frontmatter is complete: all required fields (`id`, `type`, `status`, `created`, `updated`,
      `tags`) are present and valid.
- [ ] `id` is globally unique (no collision with existing files).
- [ ] File name follows naming conventions (lowercase, hyphens, no special characters).
- [ ] Exactly one H1 heading, matching the `id` semantically.
- [ ] All mathematical notation uses LaTeX syntax (`$...$` and `$$...$$`).
- [ ] No backticks or Unicode used as math symbol substitutes.
- [ ] All symbols used are registered in `notation.md`.
- [ ] Prose lines are approximately 100 characters (soft limit).
- [ ] Blank lines before lists and after headings.
- [ ] Only regular hyphens (`-`) used for lists and separators. No em dashes or en dashes.
- [ ] No emojis or decorative symbols.
- [ ] `requires` lists only direct prerequisites (2-5 entries).
- [ ] `see_also` lists only focused related concepts (3-8 entries).
- [ ] File is under 250 lines. If approaching this limit, split.
- [ ] All in-body links use relative paths and resolve to existing files.

### Per-Ingest Checks

- [ ] All concepts from the source have been extracted and filed (or merged into existing files).
- [ ] `notation.md` updated with new symbols.
- [ ] `index.md` updated with new/modified entries.
- [ ] `log.md` entry appended with correct format.
- [ ] Conflicts flagged and logged per the conflict resolution protocol.
- [ ] Source notation translated to KB conventions where necessary.
- [ ] New files set to `status: complete` after verification.

### Per-Lint Checks

- [ ] All orphan pages identified and reported.
- [ ] All dead links identified and reported.
- [ ] All stale drafts identified and reported.
- [ ] Notation drift detected and reported.
- [ ] Unresolved conflicts reported.
- [ ] Oversized files reported.

---

## 13. Input-Type Handling

### LaTeX Sources

- **Challenge**: LaTeX uses custom macros, environments, and notation that may differ from KB
  conventions.
- **Strategy**: parse LaTeX structure (definitions, theorems, proofs, examples). Translate all
  notation to KB conventions. Extract each concept as a separate file. Preserve the logical
  structure (definition -> theorem -> proof) in the output files.

### PDF Sources

- **Challenge**: PDFs may contain complex layouts, figures, tables, and multi-column text.
  Mathematical content may be poorly OCR'd.
- **Strategy**: extract text and mathematical content carefully. Verify all formulas against the
  original. Treat each section or subsection as a potential source of multiple concepts. File
  figures as descriptions (not images) in the Remarks section if relevant.

### Plain Markdown

- **Challenge**: may lack structure or use inconsistent formatting.
- **Strategy**: impose the KB template structure. Extract concepts, verify notation, create
  proper frontmatter.

### Source Code Fragments

- **Challenge**: code implements algorithms but may not document the underlying theory.
- **Strategy**: use code as a reference for algorithm descriptions, not as a source of
  definitions. Link to the code in the Remarks section as an external annotation. Do not embed
  code in wiki files.

---

## 14. Formatting Rules (Non-Negotiable)

These rules are enforced across all wiki files. They are machine-checkable.

- **Typography**: use only regular hyphens (`-`) for lists and separators. Never use em dashes
  (`---`), en dashes (`--`), or other special characters. No decorative symbols or emojis.
- **Math**: all mathematical notation, both inline and block-level, must use LaTeX syntax
  (`$...$` and `$$...$$`). Do NOT use backticks or Unicode characters as substitutes for math
  symbols. Assume the output will be rendered by KaTeX or MathJax.
- **Line wrapping**: prose should aim for approximately 100 characters per line.
- **Spacing**: blank line before lists and after the preceding paragraph. Blank line between a
  heading and the content that follows it.
- **Tables**: standard Markdown table syntax with column alignment via dashes and colons.

---

## 15. Key Design Decisions

### 1. Concept Hierarchy: Metadata-Driven Dependencies

**Decision**: flat directory structure with `requires` field encoding dependencies.

**Justification**: deep directory hierarchies are brittle and hard to navigate for both humans
and LLMs. A flat structure with metadata-driven dependencies allows topological sorting for
reading order, flexible reorganization without moving files, and simpler git diffs. The `requires`
field encodes only direct prerequisites, keeping the dependency graph sparse and manageable.

### 2. Backlink Strategy: No Bidirectional Enforcement

**Decision**: `see_also` is unidirectional. No requirement for reciprocal links.

**Justification**: bidirectional links cause the scalability problem observed in the old
knowledge base, where `index.md` listed every file in its `related` field. With unidirectional
`see_also` and a curated `index.md`, discoverability is maintained without combinatorial
explosion. Lint operations detect missing cross-references when they matter.

### 3. Atomicity Boundary: One Definition Per File

**Decision**: one file = one primary definition + immediate supporting material.

**Justification**: in mathematics, definitions often depend on other definitions. The `requires`
field captures these dependencies explicitly. A file is atomic if it can be understood after
reading its prerequisites. Propositions are split out when they are substantial or
cross-referenced, keeping concept files focused.

### 4. Proposition Placement: Size and Reference Count

**Decision**: inline if < 20 lines and single-reference; separate file otherwise.

**Justification**: inlining short, local results keeps related material together and reduces
file proliferation. Separating substantial proofs or widely-referenced results avoids duplication
and keeps concept files under the size threshold.

### 5. Conflict Resolution: Flag-and-Append

**Decision**: contradictory information is flagged, appended, and logged for human review.

**Justification**: silent replacement risks losing valid alternative formulations. Flag-and-append
preserves both versions, annotates the discrepancy, and defers the final decision to a human
reviewer. The `log.md` entry ensures the conflict is visible during lint operations.

### 6. Executor Instructions: Unified in This Document

**Decision**: all executor agent instructions live in this document (`kb/AGENTS.md`), not in a
separate file.

**Justification**: a single canonical document prevents drift between schema rules and executor
instructions. The "Executor Agent Role" section at the top of this document provides the
operational workflow (extract, translate, create, link, verify, log), while the remaining
sections define the rules and conventions. Executors read this entire document before operating.
This eliminates duplication and ensures that updates to rules are immediately visible to
executors without requiring synchronization across multiple files.

---

## 16. Self-Containment

The `kb/` directory must function as a standalone artifact. Specifically:

- All internal links use relative paths within `kb/`.
- `notation.md`, `index.md`, `log.md`, and `template.md` are self-contained.
- References to external code (e.g., `src/skrough/`) appear only in Remarks sections as
  annotations, never as structural dependencies.
- The knowledge base can be extracted into a separate git repository and remain fully functional.

---

## 17. Source Provenance and `kb/sources/`

### Purpose of `kb/sources/`

The `kb/sources/` directory holds **source-summary** files -- bridge documents between raw source
material and the extracted wiki content. Source-summaries serve two complementary roles:

1. **Provenance metadata**: document what the source is (title, author, type, original location),
   what was extracted from it, and how to locate the original material.
2. **Distilled knowledge**: capture cross-cutting insights, methodologies, or patterns from the
   source that are not themselves wiki content (e.g., proof techniques, general frameworks,
   structural overviews).

Source-summaries are **not** raw source dumps. They are curated distillations that help future
agents and human readers understand where the knowledge came from and what methodology or context
surrounds it.

**Example**: `sources/thesis-phd.md` documents the PhD thesis (provenance) and summarizes what
was extracted. `sources/erickson-np-hardness-methodology.md` distills a general methodology for
NP-hardness proofs from an external textbook -- this methodology is not a rough set concept but
is valuable knowledge that informs how propositions in the KB were proven.

### The `source` Field Convention

Every wiki page (concept, proposition, example) that was extracted from a source must reference
the corresponding source-summary's `id` in its `source` frontmatter field. For example:

```yaml
source: src-thesis-phd
```

**Never** use external file paths (e.g., `tmp/phd/thesis.tex`) in wiki page `source` fields.
External paths are fragile -- the source files may be moved or deleted. The source-summary file
preserves the provenance information within the KB itself.

Source-summary files themselves use the `source` field differently: they store the original
external path or bibliographic citation, since they are the canonical record of where the
knowledge originated:

```yaml
# In a source-summary file:
source: "tmp/phd/thesis.tex"  # or a bibliographic citation
```

### Creating Source-Summaries

When ingesting a new major source (a book, thesis, paper, or substantial document):

1. Create a source-summary file in `kb/sources/` with `type: source-summary`.
2. Include: title, author, year, type, original path/citation, and a structural overview of what
   was extracted.
3. Optionally distill cross-cutting methodology or patterns that do not fit into individual wiki
   pages.
4. Set `source` to the original path or citation.
5. All wiki pages extracted from this source reference the source-summary's `id` in their `source`
   field.

For minor sources (a single pasted excerpt, a short text fragment), a source-summary is optional.
The `source` field can hold an inline citation string directly. However, if multiple wiki pages
are extracted from the same minor source, a source-summary should be created.

### External Knowledge Provided by the Operator

When the operator provides external knowledge as pasted text (e.g., a methodology excerpt from a
textbook, a paper section) alongside the main source material:

- Create a source-summary file in `kb/sources/` if the content has standalone value (methodology
  patterns, proof techniques, general frameworks).
- Use author-year naming: `erickson-np-hardness-methodology.md`, `pawlak-1982.md`.
- The source-summary captures the distilled knowledge; wiki pages that applied it link via
  `see_also`.

---

## 18. Ingestion Tracking

The file `kb/ingestion.md` tracks the progress of source ingestion. It serves as:

1. **Progress tracker**: checklists of items extracted from a source, marked as completed.
2. **Decision log**: records of choices made during ingestion (e.g., whether to inline or split a
   proposition, how to handle a proof gap).
3. **Session handoff**: instructions for resuming work across sessions, including pending items
   and current state.
4. **Session reflections**: patterns that worked well, caveats, and lessons learned.

### When to Use `ingestion.md`

- **Before starting an ingest**: add a new section for the source with a checklist of items to
  extract.
- **During ingest**: check off completed items, note decisions and pending work.
- **After ingest**: record final state (file counts, notation symbols, total pages).
- **Between sessions**: update the "How to resume" section with current state and next steps.

### Relationship to Other Files

- `ingestion.md` is a **working document**, not a schema rule. It changes frequently during
  ingestion and stabilizes after.
- Permanent rules and conventions belong in `AGENTS.md` (this file).
- Per-source provenance belongs in `kb/sources/`.
- Per-operation records belong in `kb/log.md`.
- When general guidelines emerge from ingestion sessions (e.g., proof handling patterns,
  verification strategies), they should be promoted to `AGENTS.md` Section 19 and removed from
  `ingestion.md`.

---

## 19. Content Extraction Guidelines

These guidelines emerged from practical ingestion experience and apply to all future source
processing.

### Proof Preservation

Preserve proofs faithfully in terms of **completeness**, not literal wording:

- No gaps, no skipped cases, no hand-waving. All branches must be checked, all non-trivial steps
  justified.
- "It is obvious" or "it follows directly" are acceptable when the step genuinely follows from a
  definition or prior result without additional reasoning -- but never when the step requires a
  non-trivial argument.
- When the source proof is detailed and step-by-step, preserve that level of detail.
- When the source proof cites an external source (e.g., "See Skowron & Rauszer 1992"), keep the
  citation but add explanatory commentary about the construction's intuition.
- If the source proof contains errors, flag them and correct in the KB (correctness >
  faithfulness).

### Proof Strategy Sections

For complex proofs (e.g., NP-hardness reductions, multi-step constructions), add an explicit
`## Proof Strategy` section before `## Proof`. This makes the reasoning structure immediately
clear to both LLM and human readers.

### Example Handling

- Small examples (single table, brief illustration): inline in the concept file's `## Example`
  section.
- Complex examples (multi-table, full dataset walkthrough): standalone file in `kb/examples/`.
- Faithfully reproduce source data line by line. Never summarise counts or invent sets when
  condensing tables -- prefer completeness over brevity.

### Citation Verification

Always verify citation titles against the source's bibliography file. Do not invent or paraphrase
paper titles, author names, or publication venues.

### Verification Patterns

Apply **three-pass verification** to each extracted item:

1. Check the statement matches the source label/reference.
2. Verify each logical step has a justification.
3. Stress-test edge cases (e.g., empty sets, boundary indices).

For NP-hardness proofs specifically:

- Verify that the cost function used in the reduction is the intended one, not the raw
  construction size.
- Distinguish "minimal" from "any" in dominating set reductions: step (=>) works for any set;
  minimality is only needed in step (<=).
- When the source says "proof is analogous", explicitly verify that every referenced lemma has
  the claimed counterpart.

### Cross-Checking

When a previous knowledge base or reference material exists, compare extracted content against it.
Flag discrepancies. This catches transcription errors and notation mismatches early.
