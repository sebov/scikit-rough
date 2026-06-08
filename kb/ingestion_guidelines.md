# Ingestion Guidelines

Universal guidelines for knowledge extraction from sources. These apply to all source ingestion and proof construction, regardless of the specific source being processed.

---

## Source Provenance and `kb/sources/`

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

## Proof Preservation

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

## Proof Strategy Sections

For complex proofs (e.g., NP-hardness reductions, multi-step constructions), add an explicit
`## Proof Strategy` section before `## Proof`. This makes the reasoning structure immediately
clear to both LLM and human readers.

## Citation Verification

Always verify citation titles against the source's bibliography file. Do not invent or paraphrase
paper titles, author names, or publication venues.

## Verification Patterns

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

## Example Handling

- Small examples (single table, brief illustration): inline in the concept file's `## Example`
  section.
- Complex examples (multi-table, full dataset walkthrough): standalone file in `kb/examples/`.
- Faithfully reproduce source data line by line. Never summarise counts or invent sets when
  condensing tables -- prefer completeness over brevity.

## Cross-Checking

When a previous knowledge base or reference material exists, compare extracted content against it.
Flag discrepancies. This catches transcription errors and notation mismatches early.

## Notation Consistency Protocol

1. **Before creating a new file**: check `notation.md` for existing symbol definitions.
2. **When ingesting a source**: translate the source's notation to match the knowledge base
   conventions. If the source uses $X$ where the KB uses $U$, rewrite as $U$.
3. **When introducing a new symbol**: add it to `notation.md` with its name, description, and the
   `id` of the file where it first appears.
4. **When a source uses genuinely new notation** (not a rename of an existing symbol): add it to
   `notation.md` and use it consistently in all new files.
5. **Conflict**: if two sources use different symbols for the same concept, the first-ingested
   convention wins. The second source's notation is noted in `notation.md` as an alternative but
   not used in wiki content.

## Conflict Resolution Protocol

When ingesting a new source, the agent may encounter claims that contradict existing wiki
content (e.g., a different definition, a conflicting theorem, updated terminology).

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
4. **Status**: set the file's `status` to `draft` if the contradiction is unresolved, signaling
   that human review is needed.

### Terminology Conflicts

If a source uses a different name for an existing concept:

- Keep the existing name as the primary name.
- Add the alternative name in the Remarks section: "Also known as <alternative name> in
  <source>."
- Do NOT create a duplicate file.

## Incremental Update Strategy

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
