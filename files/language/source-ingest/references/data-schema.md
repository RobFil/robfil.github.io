# Data Schema

## Directories

- `input/pdfs/`: original PDF sources.
- `input/text/`: manually exported or OCR-generated text files.
- `output/machine/`: JSON and JSONL files for later programmatic training workflows.
- `output/human/`: Markdown files for review, correction, and debugging by humans.
- `output/debug/`: warnings, extraction logs, skipped pages, and uncertain chunks.

## Source Index and Review States

Use `output/machine/parsed_files_index.json` as the authoritative registry. Use the source filename as key and store at least:

- `source_id`: stable identifier derived from the UTF-8 filename.
- `sha256`: content hash used to detect new file versions.
- `document_date`: ISO date inferred from a valid leading `YYYYMMDD` filename prefix or confirmed during review; otherwise `null`.
- `topic_group`: normalized thematic series title. Repeated documents may share it; a standalone PDF uses its own title.
- `document_role`: source role such as `content`, `material`, `answer`, `exercise`, or `part-N`.
- `status`: `pending_review`, `accepted`, `needs_retry`, `needs_manual_review`, `extraction_failed`, or `source_missing`.
- `attempts`: number of deterministic extraction attempts for the current indexed source.
- `artifact_jsonl`, `review_markdown`, and `review_images`: paths used for comparison and review.
- `review_history`: timestamped AI decisions and evidence notes.
- `excluded_pages`: reviewed 1-based page numbers omitted from accepted context, normally because they contain only secondary exercises with unreliable OCR or redundant material.
- `exclusion_reason`: required human-readable reason when `excluded_pages` is non-empty.

`source_id` is derived from the UTF-8 filename and connects the source JSONL, review Markdown, page-image directory, and index entry. It is not the content hash. Use `sha256` to detect identical or changed file content.

Never add `pending_review` content to the reviewed context. Rebuild `output/machine/context.jsonl` and `output/human/context.md` exclusively from entries with `status: accepted`, omitting every record whose page is listed in that source's `excluded_pages`. Preserve excluded records in the per-source JSONL for provenance.

Group accepted context by `topic_group`. Order groups by their earliest `document_date`; order sources inside each group by date and then filename. Place undated groups and sources after dated ones. Treat inferred grouping as a reviewable suggestion, not as semantic ground truth.

## Machine Records

Use JSONL for extracted chunks:

```json
{"id":"source-stem-p0001-0001","source_file":"lesson.pdf","page":1,"theme":"grammar","kind":"extracted_text","text":"...","language_hint":"ja","generated":false}
```

Required fields:

- `id`: stable id based on source stem, page, and chunk index.
- `source_file`: original filename.
- `page`: 1-based PDF page number, or `null` for non-paginated sources.
- `document_date`, `topic_group`, and `document_role`: reviewed ordering and grouping metadata copied from the source index.
- `theme`: one of `vocabulary`, `kanji`, `expressions`, `grammar`, `dialogues`, `readings`, `exercises`, `notes`, or `uncategorized`.
- `kind`: `extracted_text`, `vocabulary_item`, `grammar_point`, `kanji_item`, `dialogue`, `exercise`, or `generated_training_item`.
- `text`: extracted or generated content.
- `language_hint`: usually `ja`, `de`, `en`, or `mixed`.
- `generated`: `false` for source extraction, `true` for trainer content produced from sources.
- `review_status`: `pending` in source artifacts and `accepted` in the central reviewed context.

Recommended extraction metadata:

- `extraction_method`: `pdf_text` for an embedded text layer or `ocr` for locally recognized page images.

Recommended optional fields:

- `reading`: kana reading for vocabulary or kanji.
- `meaning`: translation or explanation from the source.
- `level`: lesson, chapter, JLPT level, or custom level when known.
- `tags`: short thematic tags.
- `confidence`: `high`, `medium`, or `low` for automatic classification.
- `notes`: extraction caveats or review notes.

## Human Review Files

Maintain `output/human/review-index.md` as the user-facing map from original filenames to source IDs, statuses, review files, and page images. Start every source review Markdown with its original filename, source ID, document metadata, extraction method, attempt count, current status, artifact links, and review note. Then list extracted pages with page metadata:

```markdown
## lesson.pdf, page 1

テーマ: grammar

...
```

Mark uncertain extraction with `TODO(review)` and keep the original text nearby. Treat review files as staging artifacts; only the central context files contain accepted learning context.

## Vocabulary Catalogue and Learner State

Store the rebuildable unique catalogue in `output/machine/vocabulary_catalog.jsonl`. Its uniqueness key is the normalized `lemma`, `reading`, and coarse `part_of_speech`, not surface spelling alone. Recommended fields are:

- `id`, `lemma`, `reading`, `part_of_speech`, and observed `forms`
- `translations`: object keyed by language code; `en` is required and contains one or more translation records, while `de` and other languages are optional
- `tags`: any compatible combination of `N3`, `N2`, `N1`, `common`, and `business`
- `jlpt_level`: `N3`, `N2`, `N1`, or `null`
- `jlpt_source`: reference name and version; required when `jlpt_level` is set
- `common`: boolean plus `common_source`
- `business`: boolean plus `business_source`, such as `reference` or `reviewed_context`
- `context_count`, `exercise_count`, and source/page provenance
- `base_priority` and a transparent breakdown of its deterministic factors

Each translation record contains:

- `text`: one meaning or sense in the language named by the parent key
- `source`: source/context identifier or external dictionary name and version
- `generated`: `false` for a reference-backed translation and `true` for an AI-created translation
- `review_status`: `accepted` or `pending`; generated translations start as `pending`

Example:

```json
{"id":"vocab-仕様-しよう-noun","lemma":"仕様","reading":"しよう","part_of_speech":"noun","translations":{"en":[{"text":"specification","source":"external_dictionary:VERSION","generated":false,"review_status":"accepted"}]},"tags":["N2","common","business"]}
```

Keep distinct senses as separate records. Require at least one accepted English translation before publishing an entry to the learner-facing catalogue. Do not infer or generate German during the deterministic run; omit `de` unless German was explicitly supplied or requested and reviewed.

Do not put learner-dependent values into the rebuildable catalogue. Store them in `output/machine/vocabulary_progress.json`, keyed by vocabulary ID, with at least:

- `mastery_evidence`: count and timestamps of known, unknown, or uncertain answers
- `manual_adjustment`: signed learner-controlled priority change
- `effective_priority`: derived from catalogue priority, mastery evidence, and manual adjustment
- `updated_at`

Exclude elementary vocabulary through a documented filter rather than deleting source occurrences. The accepted context remains unchanged and continues to provide provenance.
