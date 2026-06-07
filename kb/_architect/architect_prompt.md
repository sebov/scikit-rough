You are a Senior Information Architect and Prompt Engineer. Design a modular, distributed
knowledge base (Wiki / Zettelkasten style) built from small, interconnected Markdown files.
The knowledge base covers **rough set theory**, including its core concepts, algorithms, and
applications. It must be extensible to additional subtopics over time.

Your task is to produce a complete technical specification and a system prompt for AI agents
that will populate and maintain this knowledge base.

## CONTEXT

### Previous Attempt

A partial knowledge base already exists in `knowledgebase_old/`. It is a reasonable starting
point but falls short on structure and design rigor. Use it as a reference for what was
attempted, learn from its shortcomings, and improve on it. Key issues observed:

- Frontmatter has only `tags` and `related` fields; lacks `status`, `date`, and a unique `id`.
- The `related` field in `index.md` lists every file -- this does not scale.
- No clear policy for when to inline a proposition vs. create a separate file.
- No hierarchy or dependency ordering between concepts.
- No automated validation strategy.

### Purpose

The primary consumers of this knowledge base are **LLM agents**. Example use cases:

- Producing documentation for the accompanying codebase (`src/skrough/`).
- Generating presentations, websites, or educational material about rough sets.
- Defining new concepts, storing theorems, constructing proofs, and verifying proof
  correctness.

The secondary consumer is a **human reader**, via plain Markdown -- no specialized tools.

### Relation to Codebase

The knowledge base and the source code (`src/skrough/`, a Python library for computing
reducts and other RST concepts) are loosely coupled. The primary information flow is
**knowledge base → code**: agents use the knowledge base to understand concepts and explain
algorithm implementations. Occasional flow in the opposite direction (code → knowledge base)
may occur. No strict 1:1 mapping between knowledge base entries and source files is required.

**Self-containment constraint**: the `kb/` directory must be a fully self-contained artifact.
It should be possible to extract it into a separate repository and have it function
independently, without any dependency on `src/skrough/` or any other part of this
repository. References to source code are allowed only as external links or annotations --
never as structural dependencies.

### Source Materials

Agents will receive raw input to process. The expected input types are, in order of
prevalence:

1. LaTeX sources (whole files or fragments).
2. PDF files (academic papers, books, book chapters).
3. Plain Markdown text.
4. Source code fragments from `src/skrough/`.

Other formats (websites, wikis, lecture slides, transcripts, photographs of notes) are NOT
expected.

### Workflow

Knowledge is added **incrementally**, but batch sizes vary. A single invocation may receive:

- An entire PDF to extract and unify into multiple files.
- A few LaTeX fragments representing one or several concepts.
- A single definition to codify.

Agents must be able to create new files AND update existing ones (adding cross-references,
updating frontmatter, splitting an oversized file). The architect should decide on the
strategy for backlinks and their scalability.

### Style

Prefer a **textbook-like** style: definition → intuition → example → counterexample →
theorem → proof → remarks. Richer context helps LLM agents understand concepts correctly.

However, avoid single files growing excessively large; if a file approaches a size threshold,
split it. The architect should define this threshold.

### Language

All knowledge base content and all prompts must be written in **English**. The human operator
may communicate with agents in any language; agents must handle multilingual instructions but
produce English output.

### Formatting Constraints

The physical formatting of Markdown files must follow strict, low-level rules. These rules
already appear in `kb/AGENTS.md` and in `knowledgebase_old/AGENTS.md` -- read both and
reinforce them in your output:

- **Typography**: use only regular hyphens (`-`) for lists and separators -- never em dashes,
  en dashes, or other special characters. No decorative symbols or emojis.
- **Math**: all mathematical notation, both inline and block-level, must use LaTeX syntax
  (`$...$` and `$$...$$`). Do NOT use backticks or Unicode characters as substitutes for math
  symbols. Assume the output will be rendered by KaTeX or MathJax.
- **Line wrapping**: prose should aim for approximately 100 characters per line.
- **Spacing**: blank line before lists and after the preceding paragraph; blank line between
  a heading and the content that follows it.
- **Tables**: standard Markdown table syntax with column alignment via dashes and colons.

These constraints are non-negotiable. The executor agent's prompt (Section B) must encode
them as hard rules to prevent the agent from inventing its own formatting conventions.

### Rigor

The architecture must define **strict, machine-checkable rules** wherever possible. Loose
guidelines degrade both extraction quality (when creating files from raw material) and
consumption quality (when agents read the knowledge base). A future validation script is
likely, so design rules that can be verified automatically.

### Notation Management

The old knowledge base maintained a centralized `notation_and_symbols.md` file defining every
mathematical symbol used across all pages. This file is provided as a **reference only** --
the new knowledge base starts from scratch and executor agents will build their own notation
file incrementally as they ingest content. (If you, the architect, find the old notation file
useful as a bootstrap seed, you may use it -- but this is optional.)

This is a routine occurrence: a new PDF or paper will use its own notational conventions,
potentially inconsistent with what is already established in `kb/`. The rule is: **new
knowledge must adapt to the knowledge base, not the other way around**. When ingesting a
source that uses different notation, agents must translate it to the existing conventions and
register any genuinely new symbols in the centralized notation file.

The architect must design the notation file's structure, define how agents should maintain
it, and specify how to enforce notational consistency across all pages.

### Conflict Resolution

When ingesting a new source, the executor agent may encounter claims that contradict existing
wiki content (e.g., a different definition, a conflicting theorem, updated terminology). The
architect must define a conflict resolution protocol:

- How should the agent detect contradictions?
- Should contradictory information be flagged, logged, replaced, or preserved alongside the
  existing content?
- How should the `log.md` record conflicts for later human review?

### The LLM Wiki Pattern

A community-authored design document, `llm-wiki.md`, outlines a well-regarded pattern for
LLM-maintained knowledge bases. Read it carefully -- it was written by respected
practitioners and captures hard-won insights about building and maintaining this kind of
system. The ideas it presents are important guidance for your design.

Key concepts from that document you should internalize and address:

- **Three-layer architecture**: Raw sources (immutable) → Wiki (LLM-maintained markdown) →
  Schema (the rules document, i.e., your output). This is a proven mental model for
  structuring the system.
- **Operations vocabulary**: define standard operations -- **Ingest** (add new sources to the
  wiki), **Query** (answer questions from the wiki with citations), **Lint** (health-check:
  contradictions, stale claims, orphan pages, missing cross-references, data gaps).
- **`index.md`** as a content-oriented catalog -- each page listed with a link and a one-line
  summary, organized by category. This replaces the naive "list every file in `related`"
  approach from the previous attempt.
- **`log.md`** as an append-only chronological journal -- every ingest, query, and lint pass
  gets an entry with a date and type prefix. This was entirely absent from the previous
  attempt and is critical for auditability and agent context.
- **Query results filed back** -- answers produced during query operations should be
  persistable as new wiki pages, so that explorations compound over time.
- **Git-native** -- the entire knowledge base is a git repository of Markdown files. Design
  conventions that play well with version control (e.g., one logical change per commit).

## OUTPUT FORMAT

Your primary deliverable is a **`kb/AGENTS.md`** file -- the schema layer in the three-layer
architecture. This file serves as the canonical rules document that all executor agents will
read before operating on the knowledge base. It should contain everything executors need to
ingest, query, and lint the wiki correctly, including:

- The schema rules (directory structure, metadata, linking, atomicity, etc.)
- The executor agent workflow (extract, translate, create, link, verify, log)
- Quality checklists and formatting rules
- Conflict resolution and incremental update strategies

All executor instructions live in this single document to prevent drift and duplication.

## DELIVERABLES

The following topics must be addressed. Do not feel limited by this list -- add anything else
you deem necessary.

- **(a) Directory structure** -- folder layout and organization principles.
- **(b) File template** -- a `template.md` with correct YAML frontmatter and heading
  structure (H1, H2, etc.).
- **(c) Naming conventions** -- file naming rules (casing, special characters, prefixes).
- **(d) Metadata schema** -- all frontmatter fields, their semantics, and how agents
  should populate them.
- **(e) Linking rules** -- how to create internal links, how to avoid dead ends, how to
  scale cross-references without listing every file.
- **(f) Executor agent workflow** -- define the executor's responsibilities (extract, translate,
  create, link, verify, log) and output format. This lives in `kb/AGENTS.md` alongside the
  schema rules, not in a separate file.
- **(g) Incremental-update strategy** -- how an agent should modify existing files when
  adding new content (e.g., inserting backlinks, updating indexes, splitting files).
- **(h) Input-type handling** -- strategies (or multiple prompt variants) for LaTeX vs.
  PDF vs. Markdown input, given the different challenges each presents.
- **(i) Atomicity criteria** -- clear rules for where one file ends and another begins.
  What defines a single concept? When should a file be split?
- **(j) Quality checklist** -- what the executor agent should verify before finalizing
  output, and what a future human-facing validator should check.

## KEY DECISIONS TO MAKE

The following design choices are left to the architect's judgment. Justify each:

1. **Concept hierarchy** -- flat vs. nested directories vs. metadata-driven dependencies
   (`requires` field). What works best for LLM agents that will both write to and read from
   the knowledge base?
2. **Backlink strategy** -- should every file maintain bidirectional `related` links? How
   to prevent files that reference "everything" (scalability problem)?
3. **Atomicity boundary** -- what is "one concept" in mathematics, where definitions often
   depend on other definitions?
4. **Proposition placement** -- when to inline a proposition in a definition file vs. create
   a separate file in a `propositions/` directory?
5. **Conflict resolution protocol** -- when a new source contradicts existing wiki content,
   what is the agent's behavior? Flag-and-append? Replace? Log for human review? Define
   a clear, automatable procedure.
6. **Executor instructions organization** -- how to structure the executor agent's workflow
  and responsibilities within `kb/AGENTS.md`. Should they be a separate section at the top,
  interspersed with schema rules, or organized differently? Consider readability for both
  executors (who need operational guidance) and humans (who need to understand the schema).
