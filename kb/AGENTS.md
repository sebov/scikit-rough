# Knowledge Base Architecture and Agent Protocol

This document is the **schema layer** for the rough set theory knowledge base. It defines the
structure, conventions, and workflows that all agents must follow when ingesting sources,
answering queries, proving new results, or maintaining the wiki.

The knowledge base is a modular, distributed collection of small, interconnected Markdown files
covering rough set theory (RST). It is designed for two audiences:

1. **LLM agents** (primary): producing documentation, generating educational material, defining
   concepts, storing theorems, constructing proofs, and verifying correctness.
2. **Human readers** (secondary): browsing via plain Markdown without specialized tools.

---

## 1. Directory Structure

The knowledge base uses a shallow, category-based directory layout. Organization is driven by
metadata (the `type` field in frontmatter), not by deep folder hierarchies.

```
kb/
  AGENTS.md                 -- This file. The schema layer and agent instructions. Read before any operation.
  index.md                  -- Content-oriented catalog of all wiki pages (updated on every ingest).
  notation.md               -- Centralized registry of mathematical symbols and notation conventions.
  ingestion_guidelines.md   -- Universal guidelines for knowledge extraction (proof handling, verification).
  template.md               -- File template with correct frontmatter and heading structure.
  concepts/                 -- Core definitions, concepts, and foundational material.
  propositions/             -- Theorems, lemmas, propositions with substantial proofs.
  examples/                 -- Worked examples, counterexamples, illustrative datasets.
  sources/                  -- Source-summary files: provenance metadata + distilled key insights.
  queries/                  -- Filed query results that compound over time.
  ingestion/                -- Per-source ingestion tracking (one file per source).
```

> **Note for agents**: you do not need to read all files on startup. Read `AGENTS.md` and `index.md`
> first. Consult `ingestion_guidelines.md` only when ingesting a new source. Consult files in
> `ingestion/` only when resuming work on a specific source or checking source-specific decisions.

---

## 2. File Template

All wiki pages must follow the template defined in `kb/template.md`:

- **Required frontmatter fields**: `id`, `type`, `status`, `created`, `updated`, `tags`
- **Optional frontmatter fields**: `requires`, `see_also`, `source`
- **Heading structure**: H1 (title), H2 (Definition, Intuition, Example, Theorem, Proof, Remarks),
  H3 (subsections)

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
  agent must check for collisions before creating a new file.

---

## 4. Metadata Schema

All frontmatter fields and their semantics. See `kb/template.md` for the complete template.

### Required Fields

- **`id`**: Globally unique identifier. Format: `<type-prefix>-<slug>`. Type prefixes:
  `concept-`, `prop-`, `ex-`, `src-`, `query-`. Slug matches the file name (without extension).
- **`type`**: One of: `concept`, `proposition`, `example`, `source-summary`, `query-result`.
- **`status`**: One of: `draft`, `complete`, `reviewed`, `deprecated`. Lifecycle: `draft` ->
  `complete` -> `reviewed` -> `deprecated`.
- **`created`**: Date the file was first created (YYYY-MM-DD).
- **`updated`**: Date of the most recent modification (YYYY-MM-DD).
- **`tags`**: Categorization keywords. Use controlled vocabulary: `core`, `reduction`,
  `approximation`, `rules`, `evaluation`, `consistency`, `complexity`, `bireducts`,
  `positive-region`, `indiscernibility`, `decision-table`. Add new tags only when no existing
  tag fits.

### Optional Fields

- **`requires`**: Prerequisite concepts (direct only, not transitive). Keep short (2-5 entries).
  Enables topological sorting for reading order.
- **`see_also`**: Related concepts for further reading (horizontal links). Keep focused (3-8
  entries). Unidirectional -- no requirement for reciprocal links.
- **`source`**: For wiki pages (concepts, propositions, examples), this field holds the `id` of
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
- Orphan detection is part of the Lint operation (see Section 9).

---

## 6. Atomicity and Proposition Placement

A single file covers **one primary mathematical concept** -- one definition and its immediate supporting material (intuition, example, basic properties).

### Inline vs. Separate File

**Inline** when: short (< 20 lines including proof), directly supports the definition, and would be orphaned as standalone.

**Separate file** when: substantial proof (>= 20 lines), referenced from multiple concepts, complex example, or multiple distinct formulations.

### Proposition File Structure

Proposition files use the same template but emphasize: **Background** (context), **Statement** (formal proposition), **Proof** (complete), **Consequences** (if any).

### File Size

**Target**: 100-200 lines per file.

---

## 7. Notation Management

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

All mathematical notation uses LaTeX syntax (`$...$` for inline, `$$...$$` for block). Never use
backticks or Unicode characters as substitutes for math symbols. Assume output will be rendered by
KaTeX or MathJax.

---

## 8. Operations

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
4. Translate the source's notation to match `kb/notation.md` conventions.
5. Update `notation.md` with any new symbols.
6. Update `index.md` with new or modified entries.
7. Set all new files to `status: complete` after verification (see Quality Checklist, Section 10).

**Output Format**: For each file you create or modify, output the complete file content including
frontmatter. Clearly label each file with its path (e.g., `kb/concepts/indiscernibility.md`).
After all files are produced, output updated `notation.md` entries and updated `index.md` entries.

### Query

**Trigger**: a question is asked against the knowledge base.

**Procedure**:

1. Read `index.md` to identify relevant pages.
2. Read the relevant pages.
3. Synthesize an answer with citations to specific file `id`s and section headings.
4. If the answer is substantial or reusable, file it as a new page in `queries/` with
   `type: query-result`.

### Prove

**Trigger**: a new theorem, proposition, or conjecture needs to be proven or verified.

**Procedure**:

1. Read relevant concept and proposition files to understand the context and existing results.
2. Check `notation.md` for symbol conventions.
3. Construct the proof following the guidelines in `kb/ingestion_guidelines.md`.
4. If the proof is substantial (>= 20 lines) or references multiple concepts:
   - Create a new file in `propositions/` with `type: proposition`.
   - Update the relevant concept files with inline summaries and links.
5. If the proof is short and only relevant to one concept:
   - Add it inline to the relevant concept file under `## Proposition` or `## Theorem`.
6. Verify the proof using the three-pass verification pattern (`kb/ingestion_guidelines.md`).
7. Update `index.md`.

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

---

## 10. Incremental Update Strategy

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

### Backlink Updates

- When file A is created and references file B (in `requires` or `see_also`), do NOT add a
  reciprocal link in file B.
- File B will discover file A through `index.md` or through lint operations that detect missing
  cross-references.
- This prevents the scalability problem of the old knowledge base where every file listed every
  other file.

---

## 11. Quality Checklist

The agent must verify the following before finalizing any output:

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
- [ ] Conflicts flagged per the conflict resolution protocol.
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

## 12. Input-Type Handling

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

## 13. Formatting Rules (Non-Negotiable)

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

## 14. Ingestion Tracking

Ingestion tracking is split into two parts:

### Universal Guidelines: `kb/ingestion_guidelines.md`

Contains guidelines that apply to **all** source ingestion, regardless of the specific source:

- Proof preservation (completeness over literal wording)
- Proof strategy sections for complex proofs
- Citation verification
- Three-pass verification patterns
- Example handling (inline vs. standalone)
- Cross-checking against reference material

These guidelines are stable and rarely change. New agents should read this file before their first
ingest operation.

### Per-Source Tracking: `kb/ingestion/`

Each source has its own file in `kb/ingestion/` (e.g., `thesis-phd.md`, `pawlak-1982.md`). Each
file serves as:

1. **Progress tracker**: checklists of items extracted from the source, marked as completed.
2. **Decision log**: records of choices made during ingestion (e.g., whether to inline or split a
   proposition, how to handle a proof gap).
3. **Session handoff**: instructions for resuming work across sessions, including pending items
   and current state.
4. **Session reflections**: patterns that worked well, caveats, and lessons learned.

#### When to Use Per-Source Files

- **Before starting an ingest**: create a new file in `ingestion/` with a checklist of items to
  extract.
- **During ingest**: check off completed items, note decisions and pending work.
- **After ingest**: record final state (file counts, notation symbols, total pages).
- **Between sessions**: update the "How to resume" section with current state and next steps.

#### Relationship to Other Files

- Per-source files in `ingestion/` are **working documents**, not schema rules. They change
  frequently during ingestion and stabilize after.
- Universal guidelines belong in `ingestion_guidelines.md`.
- Permanent rules and conventions belong in `AGENTS.md` (this file).
- Per-source provenance belongs in `kb/sources/`.
- When general guidelines emerge from ingestion sessions, they should be promoted to
  `ingestion_guidelines.md` and removed from session-specific notes.
