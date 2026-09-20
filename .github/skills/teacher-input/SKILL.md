---
name: teacher-input
description: Prepare Japanese learning PDFs and text exports as reviewed local teaching context. Use for importing, OCR, source indexing, provenance, and context rebuilds; do not use for writing reading texts or conducting lessons.
---

# Japanese Teacher Input

Turn local learning material into a reliable corpus for `teacher-write` and
`teacher-read`.

## Scope

- Store raw PDFs in `files/language/source-ingest/input/pdfs/` and supplied text
  exports in `files/language/source-ingest/input/text/`.
- Use `files/language/source-ingest/scripts/manage_extractions.py` for normal
  extraction and rebuilds. Read its data schema before changing output formats.
- Preserve original filename, page provenance, topic group, document role and
  extraction status.
- Rebuild `output/machine/context.jsonl` only from accepted sources. Treat it,
  together with `parsed_files_index.json`, as the teaching contract.

## Boundaries

- Keep OCR, raw input and debug artifacts out of teaching truth.
- Record uncertain pages or extraction failures; never invent missing content.
- Do not create reading posts, conduct reading practice, or correct a learner.
- Keep generated local extraction artifacts out of Git unless the learner asks
  for a portable snapshot.
