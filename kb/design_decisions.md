# Key Design Decisions

This document captures the rationale behind major architectural choices in the knowledge base.
These decisions inform future evolution of the schema but are not required reading for routine
agent operations.

---

## 1. Concept Hierarchy: Metadata-Driven Dependencies

**Decision**: flat directory structure with `requires` field encoding dependencies.

**Justification**: deep directory hierarchies are brittle and hard to navigate for both humans
and LLMs. A flat structure with metadata-driven dependencies allows topological sorting for
reading order, flexible reorganization without moving files, and simpler git diffs. The `requires`
field encodes only direct prerequisites, keeping the dependency graph sparse and manageable.

## 2. Backlink Strategy: No Bidirectional Enforcement

**Decision**: `see_also` is unidirectional. No requirement for reciprocal links.

**Justification**: bidirectional links cause the scalability problem observed in the old
knowledge base, where `index.md` listed every file in its `related` field. With unidirectional
`see_also` and a curated `index.md`, discoverability is maintained without combinatorial
explosion. Lint operations detect missing cross-references when they matter.

## 3. Atomicity Boundary: One Definition Per File

**Decision**: one file = one primary definition + immediate supporting material.

**Justification**: in mathematics, definitions often depend on other definitions. The `requires`
field captures these dependencies explicitly. A file is atomic if it can be understood after
reading its prerequisites. Propositions are split out when they are substantial or
cross-referenced, keeping concept files focused.

## 4. Proposition Placement: Size and Reference Count

**Decision**: inline if < 20 lines and single-reference; separate file otherwise.

**Justification**: inlining short, local results keeps related material together and reduces
file proliferation. Separating substantial proofs or widely-referenced results avoids duplication
and keeps concept files under the size threshold.

## 5. Conflict Resolution: Flag-and-Append

**Decision**: contradictory information is flagged, appended, and logged for human review.

**Justification**: silent replacement risks losing valid alternative formulations. Flag-and-append
preserves both versions, annotates the discrepancy, and defers the final decision to a human
reviewer. The `log.md` entry ensures the conflict is visible during lint operations.

## 6. Agent Instructions: Unified in AGENTS.md

**Decision**: all agent instructions (schema rules, workflows, quality checklists) live in a
single `AGENTS.md` file. Supporting documents (design rationale, ingestion guidelines, per-source
tracking) are split into separate files.

**Justification**: a single source of truth for agent behavior prevents drift and duplication.
However, not all documents need to be read by every agent. The schema layer (`AGENTS.md`) is
required reading; supporting documents are consulted on demand (e.g., `ingestion_guidelines.md`
only by ingest agents, `design_decisions.md` only when understanding architectural rationale).
This balances consistency with context efficiency.

## 7. Ingestion Tracking: Split by Scope

**Decision**: universal extraction guidelines in `ingestion_guidelines.md`; per-source tracking
in `ingestion/<source>.md` files.

**Justification**: a single `ingestion.md` file grows unbounded as sources are added. Splitting
by scope allows: (1) stable guidelines that rarely change, (2) per-source files that scale
independently, (3) agents reading only what they need (query agents skip ingestion files
entirely). Each source file is self-contained with its own checklist, decisions, and session
history.
