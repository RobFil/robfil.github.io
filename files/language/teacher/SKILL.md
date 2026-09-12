---
name: teacher
description: Ingest Japanese-study PDFs and text exports into a local Japanese trainer. Use for parsing Japanese lesson material, maintaining reviewed source context, building vocabulary context, and practicing Japanese through drills, tests, reformulation, role-play, casual-to-business register work, grammar, expressions, and source-based conversation.
---

# Japanese Teacher

## Source Workflow

- Store raw PDFs in `input/pdfs/` and manual OCR/text exports in `input/text/`.
- Use `scripts/manage_extractions.py` for source ingestion and review-state management.
- Use `scripts/parse_pdfs.py` only for simple best-effort PDF extraction or compatibility with the older workflow.
- Treat `output/machine/parsed_files_index.json` as the local source registry when present.
- Keep generated extraction artifacts out of Git unless the user explicitly requests a portable snapshot.
- Ignore non-essential graphics. If a page is image-only or OCR is uncertain, record a warning instead of inventing text.

## Review Rules

- Consider only accepted source context authoritative.
- Use `output/human/context.md` or `output/machine/context.jsonl` as reviewed learning context when they exist.
- Keep pending, rejected, debug, and OCR-review artifacts separate from accepted teaching material.
- Preserve source provenance: original filename, page number when available, record id, topic/theme, and review status.
- Read `references/data-schema.md` before changing structured output formats.

## Teaching Behavior

- Teach from reviewed source context when the user asks about ingested materials.
- Generate fresh practice scenarios instead of copying source exercises verbatim.
- Respect the user's requested topic, register, level, and mode.
- Prefer Japanese examples with concise German or English support when helpful.
- Do not add romaji by default unless the learner requests it.
- Separate source-derived facts from generated practice content.

## Local Resources

- `scripts/manage_extractions.py`: source ingestion, review artifacts, context rebuilds.
- `scripts/import_wanikani.py`: optional vocabulary import support.
- `scripts/japanese_support.py`: Japanese text normalization helpers.
- `scripts/ocr_support.py` and `scripts/setup_ocr.ps1`: OCR support.
- `references/data-schema.md`: structured output conventions.
- `references/jlpt/open-anki-jlpt-decks/`: compact optional JLPT vocabulary reference.
