---
name: teacher-write
description: Create original Japanese reading texts from reviewed local learning context and publish-ready Jekyll posts. Use for guided, random, or varied reading-text generation; do not use for live reading feedback or source ingestion.
---

# Japanese Teacher Write

Create original, source-grounded Japanese reading practice for the site.

## Sources and selection

- Resolve the repository root before reading learning context: walk up from
  this skill file until both `_posts/reading/` and
  `files/language/source-ingest/` exist. The canonical input is
  `files/language/source-ingest/output/machine/context.jsonl`; the matching
  source registry is `files/language/source-ingest/output/machine/parsed_files_index.json`.
  Do not resolve this path relative to the current shell directory or the
  skill directory.
- Read teaching material only from accepted records in `context.jsonl`,
  cross-checked against accepted entries in `parsed_files_index.json`. Never
  read raw PDFs, pending records, source-side JSONL files, OCR/debug artifacts
  or review drafts as a substitute for the accepted context.
- If the canonical context or index is missing, empty, or has no eligible
  accepted material for the requested track, stop source selection and tell
  the learner the exact missing path. Do not search for PDFs or extract them
  from this writing skill; source preparation belongs to `teacher-input`.
- Honour a learner-named source, grammar focus, topic, level, length and track.
- Before selecting a grammar focus, read
  `files/language/teacher/readings/grammar-focus-history.yml`. It is the
  permanent rotation record; published post front matter is a cross-check, not
  a substitute. Prefer an eligible grammar that has never been a focus. Never
  repeat a recorded focus while an eligible unused alternative exists. Also
  read `files/language/teacher/readings/topic-history.yml` and avoid topics
  already recorded there while an eligible unused topic exists. Keep both
  registers append-only, including replaced or removed posts.
- Default to `random`: randomly select an eligible accepted source, an unused
  grammar focus and an unused topic group. Apply the learner's track and other
  constraints before randomizing. Use `guided` only when the learner names a
  source, grammar or topic; record any explicitly requested repeat and its
  reason. Use `varied` only when the learner asks for a deliberate mix or
  variation. When randomizing, also avoid either of the two latest topic groups
  if another eligible group exists.
- Keep Nana/business material to accepted `Unit` sources and Norio/grammar
  material to accepted non-`Unit` sources unless the learner explicitly asks to
  cross the tracks. Keep one text focused on one topic group unless asked to mix.

## Writing and posts

- Write original Japanese; never reproduce source passages. Use grammar
  naturally in one coherent situation with development and resolution.
- Default to a pair of N2 texts, each about 700–1,100 Japanese characters and
  ten minutes of reading time, unless the learner asks otherwise.
- Create posts in `_posts/reading/` using the schema in
  `files/language/teacher/readings/README.md`. Record selection mode and exact
  contributing filenames in front matter; use `general_n2` only when no source
  materially informed the grammar.
- After creating or changing a post, update
  `files/language/teacher/readings/grammar-focus-history.yml` and
  `files/language/teacher/readings/topic-history.yml` in the same change.
  Preserve prior entries as historical records; add each new focus and topic,
  and mark superseded entries as `replaced` or `removed`. Record the source,
  selection mode, topic and exact grammar focus so the next random selection
  can avoid immediate repetition.
- Keep the Japanese body uninterrupted and selectable: no romaji, furigana,
  embedded translation or word-by-word markup. Put the German summary and a
  compact grammar note after the text.
- For current-news practice, research a current Japanese primary source where
  possible, write an original faithful summary, and record the URL, publication
  date and check date. Mark invented news-style practice as fictional.

## Publication boundary

- Creating or editing a local post is the normal output of this skill.
- Do not commit, push, open a pull request, or otherwise publish to GitHub
  unless the learner explicitly asks.
- Do not run an interactive reading session; hand that work to `teacher-read`.
