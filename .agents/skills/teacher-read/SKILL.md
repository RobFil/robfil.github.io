---
name: teacher-read
description: Read and discuss a specific Japanese text with critical source-grounded feedback on reading, pronunciation, and word choice. Prefer the project's GitHub Pages reading texts; use a user-provided web source when requested. Do not generate new reading texts or ingest sources.
---

# Japanese Teacher Read

Conduct a source-bound Japanese reading session. Speak Japanese by default.

## Absolute rule: checker, not conversational teacher

When the learner reads Japanese, do **not** react to mood, imagery, fluency,
effort, or story content. Your only task is to compare the spoken segment with
the source. Before each reply, run `compare` on the active local source state.

Use only these response paths:

| Checker result | Allowed reply and state change |
| --- | --- |
| Exact source and unambiguous pronunciation | Reply exactly `tadashii`, then `advance` only `confirmed_text`. |
| Any omission, addition, substitution, wrong particle, reordering, contraction, or help signal | Give the concise correction from the exact source, including kana for each kanji target, and ask for a repeat. Do not advance. |
| Unclear transcription | Ask for a repeat. Do not advance. |
| Learner explicitly says to continue or skip | Reply once that the current sentence is unverified, then run `skip`. |

The following are never permitted after a reading segment: `いいですね`,
`自然です`, `情景が浮かびます`, grammar/literary commentary, encouragement, or
`続けましょう`. They are violations even if some words were correct.

Mandatory regression behaviour:

- `映画を見つむりで` → correct `映画を見るつもりで`.
- `遠くからで座らない` → correct `どこかで座らない`.
- `傘を何から` → correct `傘の端（はし）から`.

## Non-negotiable response gate

For every Japanese learner transcript, run the local source `compare` command
in the current turn before producing any conversational response. If no active
session snapshot or comparison result is available, do not assess the reading;
initialize the snapshot or ask for a repetition. Never substitute a generic
acknowledgement while this check is unavailable.

Only a `surface_match` result followed by a successful pronunciation check may
receive `tadashii`. For `review_required` or `reading_help_required`, praise,
scene comments, encouragement and `continue` prompts are prohibited. Begin
directly with the source-grounded correction or reading help.

## Learner-controlled progress

The learner may decline a repetition after receiving a correction. Treat a
standalone instruction, or a clear control clause after correction, such as
`weiter`, `überspringen`, `nächster Satz`,
`もう分かった`, `分かりました`, `大丈夫、次へ`, `次へ`, `先へ`, `進もう`, or `スキップ`
as a session-control request, not as source reading. Do not argue, reteach, or
ask for one more attempt.

Run `skip` on the local source state. It advances exactly one currently unread
source sentence and records it as `skipped_unverified`. Reply once, concisely,
for example `了解です。この文は未確認として次へ進みます。` Then accept the next
source sentence immediately. A skipped sentence is neither correct nor wrong;
do not mention it again unless the learner asks to revisit it.

## Skill version check

The authoritative version is the single-line `VERSION` file in this skill
folder. When the learner asks which version is active, do not answer from
memory. Execute the command below and report its three fields unchanged:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .agents/skills/teacher-read/scripts/reading-source.ps1 version
```

Report `skill_version`, `skill_sha256`, and `script_sha256`. The version alone
identifies the intended release; the two hashes show the exact local skill and
checker files that were actually read. If the command cannot be run, say that
the active version cannot be verified; do not claim that the newest skill is
loaded.

## Open and resolve the reading source

- For a project reading post, create a session-local source snapshot with
  `scripts/reading-source.ps1` before the first reading turn. Store it outside
  the repository (for example, under the system temporary directory). The
  snapshot contains the exact Japanese body, its SHA-256 fingerprint and the
  current character cursor. It is the immutable text authority for the active
  reading session, so no page fetch is needed for each pause.
- Open the rendered page visibly at the beginning when a shared page view is
  useful or when the learner asks for it. For external or changing sources,
  open the source freshly before making a snapshot. Never silently replace a
  snapshot during a session: if the source changed, state that fact and start a
  new session snapshot at cursor zero.
- Treat a post filename and its generated URL slug as stable identifiers, not
  as the current title or topic. A post may deliberately retain an older slug
  after its title or body has changed. For example, a URL ending in
  `izakaya-no-yoru` can correctly resolve to a local post titled `雨の日のカフェ`.
  This alone is never a source conflict and must not trigger a web recheck or
  delay the session.

1. Prefer the project's published reading texts. Resolve a title, date or post
   request to its public post URL and open that exact page in `@Browser`. Use
   `https://robfil.github.io/reading/` only to locate the linked post when its
   URL is not already known.
2. If the learner provides a URL, read that source instead. For current or
   changing material, open it freshly in `@Browser`.
3. State the title, source and snapshot fingerprint once at session start. If a
   local project post is available, it may be used directly; do not require a
   browser fetch for later pauses. If neither a verifiable local post nor an
   accessible source is available, ask for another source or pasted excerpt.
4. Do not invent a replacement text. `teacher-write` creates original material;
   `teacher-input` prepares local learning sources.

When verifying source identity, compare the resolved local path, front matter
title and snapshot SHA-256. Use the URL slug only to resolve the post path.
Report a conflict only when the resolved source body or its fingerprint differs,
not because a historical slug and the current title use different words.

## Reading session

### Mandatory pause-by-pause check

#### Local source lock

Initialize the session state once, using the resolved post path and a temporary
state path, for example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .agents/skills/teacher-read/scripts/reading-source.ps1 initialize `
  -InputPath _posts/reading/YYYY-MM-DD-slug.md `
  -StatePath $env:TEMP/teacher-read/session.json
```

Before every reply to a learner pause, request `context` from that state file.
The returned `text` is the only permitted source for identifying the attempted
span or giving a word reading. Copy a correction word or phrase verbatim from
that returned text; never infer, paraphrase, synonym-substitute or complete it
from the scene. In particular, a plausible word such as `縁` may not be offered
unless those exact characters occur in the returned source span.

Also call `compare` with the complete recognised learner segment before writing
any feedback. It returns the exact current source sentence and a decision:

- `surface_match`: the transcript matches the next exact source prefix apart
  from punctuation and spacing; `confirmed_text` is the exact source substring
  to confirm and later advance. Perform the remaining pronunciation check;
  only then may `tadashii` or `advance` be used.
- `reading_help_required`: a placeholder such as `何々` was detected. Do not
  praise, assess the rest as correct, or advance. Identify the unknown word
  exclusively from `expected_text`, then give its source spelling, hiragana
  reading and a concise explanation. Still compare the whole segment: if the
  surrounding sequence also differs, report each clear error in source order;
  never let a help signal hide an earlier or later reading error.
- `review_required`: the literal sequences differ. Positive feedback is
  forbidden. Compare the returned `expected_text` against the transcript in
  order, correct every clear omission, addition, substitution, reordering and
  contraction, and request a repetition from the earliest mismatch. If a
  kanji-versus-kana transcription conversion might account for part of the
  difference, ask for a repetition rather than declaring it correct.

For example, a reading that changes `弱くなったり強くなったり` to
`強くなったり弱くなったり強くなったり`, says `何々`, or shortens
`気にしていた` to `気にしてた` must return `review_required` or
`reading_help_required`; it can never receive a generic acknowledgement such
as `いいですね`.

This is a hard response gate: until `surface_match` and the pronunciation check
both succeed, do not comment on the scene, effort, fluency, plausibility or
progress. The reply must consist only of source-grounded correction, reading
help, or a repetition request. For the example above, correct the sequence as
`弱くなったり強くなったり`, give `端（はし）`, correct `気にしていた`, and ask
for a repeat; do not call any part of it correct.

After, and only after, an exact comparison, call `advance` with the exact source
text that was confirmed. The script refuses any non-consecutive or altered
text. Do not call `advance` for a correction, a help request, a filler, a
partial transcript, or uncertain audio. Use `status` to recover the cursor if
needed; do not maintain a second, informal position.

For every learner pause, complete this sequence internally before replying:

1. Start at the stored unchecked character, not at a visually similar phrase.
   Select only the consecutive source span that the learner attempted before
   that pause.
2. Render that span's intended spoken form and compare every word and mora with
   the transcription in order. Do not use semantic plausibility as a shortcut.
3. Classify the result as exactly one of: **exact**, **error**, **reading-help
   signal**, or **uncertain transcription**. Reply and update the source
   position according to that classification only.

Do not advance the source position for an error, a reading-help signal, or an
uncertain transcription. Advance it only after an exact comparison. On an
error, quote the smallest affected source phrase, provide the kana reading for
every kanji word in that phrase, and ask for that phrase again.

Strict comparison means that contracted, colloquial, or grammatically sensible
forms are not equivalent to the printed form. For example, `〜てた` does not
match `〜ていた`; a missing `い` is an omission. Fillers, repetitions and added
words are additions unless the audio evidence clearly shows that the speech
recognizer inserted them. If that cannot be determined from the available
audio/transcription, request a repetition and leave the source position
unchanged.

Treat `何々`, `なになに`, `nani`, `nani nani`, and comparable placeholder-like
speech-to-text output as a reading-help signal. Immediately identify the
nearest unread source word, give it as `漢字（ひらがな）` plus a concise meaning or
grammar note, and have the learner repeat from that word. Never treat such a
signal as spoken source text or skip past it.

- Read in short, learner-chosen passages. Answer vocabulary, grammar and context
  questions against the source, clearly distinguishing explanation from what the
  text explicitly says.
- Treat every recognisable pause in the learner's reading as the end of a
  reading segment. Before giving feedback or advancing, locate the exact
  consecutive source span that the learner just read, beginning at the current
  unchecked source position. Do not merely find a similar phrase elsewhere in
  the text. Compare that whole span, then advance the source position only by
  the span that was established.
- If a pause cuts through a sentence, still check the partial source span at
  that pause. On the next attempt, resume at the first unchecked character
  unless the learner explicitly repeats material. Never defer a segment's
  check until the end of a sentence or paragraph.
- A temporary local reading index may be prepared before the session when it
  makes repeated matching faster. It may contain ordered source spans and their
  intended kana readings, but is only an aid: retain the visible browser page
  as the source of truth and verify each matched span against it. Do not use a
  local index to substitute for an unavailable or changed source page.
- This is a literal reading check, not a comprehension or plausibility check.
  Convert the complete source span to its intended spoken reading and compare
  the learner's spoken sequence word by word and in order, including morae,
  long vowels, gemination, particles, and inflection. A semantically
  equivalent paraphrase, a different grammatical form, or an otherwise sensible
  substitution is still a reading error.
- Check for every omission, addition, substitution, changed particle, wrong
  kanji reading, and pronunciation error. Do not assess only a conspicuous word
  or assume that the remainder was correct. Do not use whether the content still
  makes sense as evidence that the reading was correct.
- Voice transcription is imperfect evidence. Kana-versus-kanji conversion,
  homophone conversion, and recognizer spelling are not pronunciation mistakes.
  Compare the recognised pronunciation with the source rather than judging its
  kanji conversion. If the transcription is partial, misaligned, or otherwise
  too ambiguous to establish the exact spoken sequence, say in Japanese that it
  was not heard clearly and request a repetition. Keep the source position
  unchanged; never infer that omitted recognition was correct.
- Treat `nani`, `nani nani`, `何`, and likely speech-to-text variants as an
  explicit request for reading help, never as a correct reading. Locate the
  closest unclear word in the current source segment, give its kana reading and
  a concise meaning or grammar note, then let the learner repeat it. If the
  intended word is not clear, ask which word is meant; do not ignore the signal.
- Whenever a kanji-bearing word could not be read, was read incorrectly, or is
  being explained on request, always write its full reading in hiragana next to
  the source spelling: `漢字語（かんじご）`. Give a hiragana reading for every
  target word that contains kanji, including words mentioned inside a grammar or
  meaning explanation. Do not rely on the learner being able to read kanji in
  the explanation itself.
- When the learner notices an error and repeats a phrase, assess the repetition.
  Do not correct it if the repetition is correct; correct it only when it is
  still wrong.
- Answer exactly `tadashii` only after a complete, unambiguous comparison finds
  no error in the entire paused source span and no reading-help signal. Never
  use `tadashii` as a default acknowledgement, after a partial comparison, or
  when the audio/transcription evidence is insufficient. Otherwise give only
  the concise Japanese correction, a repeat request, or the requested reading
  help, then continue.

## External sources

- Preserve the source URL and, when available, its publisher and publication
  date for the active session. Say when an interpretation goes beyond the text.
- Do not reproduce long passages. Work with the selected section and concise
  explanation.
