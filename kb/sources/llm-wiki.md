---
id: src-llm-wiki
type: source-summary
status: complete
created: 2026-06-07
updated: 2026-06-07
tags: [core, architecture]
source: "https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f"
---

# LLM Wiki Pattern (Karpathy)

A design document by Andrej Karpathy describing a pattern for building personal knowledge bases using LLMs.

## Core Concept

Instead of RAG (where LLM re-derives knowledge on every query), the LLM incrementally builds and maintains a persistent wiki -- a structured, interlinked collection of markdown files. The wiki is a compounding artifact: cross-references are pre-built, contradictions are flagged, synthesis reflects all ingested sources.

## Three-Layer Architecture

1. **Raw sources** -- immutable collection (articles, papers, data). LLM reads but never modifies.
2. **Wiki** -- LLM-generated markdown files (summaries, entity pages, concept pages). LLM owns this layer entirely.
3. **Schema** -- rules document (e.g., AGENTS.md) telling the LLM how the wiki is structured and what workflows to follow.

## Operations

- **Ingest**: process new source, extract knowledge, update wiki pages, update index, append to log.
- **Query**: answer questions from wiki, optionally file results back as new wiki pages.
- **Lint**: health-check for contradictions, stale claims, orphan pages, missing cross-references.

## Key Insights

- **index.md** as content-oriented catalog (each page with link + one-line summary) works surprisingly well at moderate scale (~100 sources, hundreds of pages).
- **log.md** as append-only chronological journal with parseable prefixes (e.g., `## [YYYY-MM-DD] ingest | Title`).
- **Git-native**: wiki is a git repo of markdown files, enabling version history and collaboration.
- Humans abandon wikis because maintenance burden grows faster than value. LLMs handle the maintenance.

## Relation to This KB

This knowledge base (`kb/`) implements the LLM Wiki pattern for rough set theory. The schema layer is `kb/AGENTS.md`. The three-layer architecture is followed: raw sources are external (PDFs, LaTeX), wiki pages are in `kb/concepts/`, `kb/propositions/`, etc., and the schema is `kb/AGENTS.md`.

## Notes

This document was originally stored in `kb/_architect/llm-wiki.md` as reference material for the architect who designed the KB schema. It was moved to `sources/` to maintain self-containment and remove the `_architect/` directory after the schema design was complete.
