# Ingestion Guidelines

Universal guidelines for knowledge extraction from sources. These apply to all source ingestion and proof construction, regardless of the specific source being processed.

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
