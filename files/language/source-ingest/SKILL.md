---
name: source-ingest
description: Ingest Japanese-study PDFs and text exports into a reviewed local source corpus. Use for parsing PDFs, OCR handling, source indexing, provenance tracking, review-state management, context rebuilds, vocabulary imports, and producing machine-readable or human-readable source artifacts. Do not use this skill for live teaching or conversation practice except to prepare the reviewed context used by a teacher skill.
---

# Japanese Source Ingest

## Purpose

Turn local Japanese learning sources into reviewed context files that other skills can teach from.

## Workflow

- Store raw PDFs in `input/pdfs/` and manual OCR or text exports in `input/text/`.
- Use `scripts/manage_extractions.py` for source ingestion, review artifacts, accepted/rejected state, and context rebuilds.
- Use `scripts/parse_pdfs.py` only for simple best-effort extraction or compatibility with older artifacts.
- Treat `output/machine/parsed_files_index.json` as the local source registry when present.
- Keep generated extraction artifacts out of Git unless the user explicitly requests a portable snapshot.
- Ignore non-essential graphics. If a page is image-only or OCR is uncertain, record a warning instead of inventing text.

## Output Rules

- Preserve provenance: original filename, page number when available, record id, topic/theme, and review status.
- Keep pending, rejected, debug, and OCR-review artifacts separate from accepted context.
- Consider only accepted source context authoritative.
- Rebuild accepted teaching context into `output/human/context.md` and `output/machine/context.jsonl` when appropriate.
- Read `references/data-schema.md` before changing structured output formats.

## Local Resources

- `scripts/manage_extractions.py`: source ingestion, review artifacts, context rebuilds.
- `scripts/import_wanikani.py`: optional vocabulary import support.
- `scripts/japanese_support.py`: Japanese text normalization helpers.
- `scripts/ocr_support.py` and `scripts/setup_ocr.ps1`: OCR support.
- `references/data-schema.md`: structured output conventions.
- `references/jlpt/open-anki-jlpt-decks/`: compact optional JLPT vocabulary reference.
