---
name: teacher-write
description: Create original Japanese reading texts from reviewed local learning context and publish-ready Jekyll posts. Use for guided, random, or varied reading-text generation; do not use for live reading feedback or source ingestion.
---

# Japanese Teacher Write

Create original, source-grounded Japanese reading practice for the site.

## Sources and selection

- Read accepted context only from
  `files/language/source-ingest/output/machine/context.jsonl` and its source
  index. Never use raw OCR, pending records or debug artifacts as learning
  truth.
- Honour a learner-named source, grammar focus, topic, level, length and track.
- Use `guided` for an explicitly named source or grammar, `random` for a
  shuffled eligible-topic deck, and `varied` by default. For `random` and
  `varied`, use published post front matter as the rotation record and avoid
  either of the two latest topic groups when an alternative exists.
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
- Keep the Japanese body uninterrupted and selectable: no romaji, furigana,
  embedded translation or word-by-word markup. Put the German summary and a
  compact grammar note after the text.
- For current-news practice, research a current Japanese primary source where
  possible, write an original faithful summary, and record the URL, publication
  date and check date. Mark invented news-style practice as fictional.

## Publication boundary

- Creating or editing a local post is the normal output of this skill.
- Do not commit, push, open a pull request, or otherwise publish to GitHub
  unless the learner explicitly asks. GitHub Pages deploys after a push.
- Do not run an interactive reading session; hand that work to `teacher-read`.
