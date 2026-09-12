# JLPT Vocabulary Reference Provenance

- Project: `jamsinclair/open-anki-jlpt-decks`
- Upstream: https://github.com/jamsinclair/open-anki-jlpt-decks
- License: MIT; see `LICENSE` in this directory.
- Upstream commit: `1ad66734417aca9dbcca6b2d5ee440cb13ab3ba0`
- Upstream commit date: `2025-08-11T03:03:02Z`
- Retrieved: 2026-07-13

## Files

| File | Data rows | SHA-256 |
| --- | ---: | --- |
| `n1.csv` | 2699 | `120911636c019899552aa6d7bd64b036ecef4bedfe272a744f75735c46aae5cd` |
| `n2.csv` | 1906 | `2d0f1ddd6222881cd9fc2ca701db74300af99b3f1f84d5ac3c18411c20f0c055` |
| `n3.csv` | 2140 | `ba071571d344e60b0e1fd11cd2e98a6aaa04515361653ca45147129460531297` |

Each CSV contains `expression`, `reading`, `meaning`, `tags`, and `guid`. The English `meaning` field is suitable as the initial catalogue translation.

## Reliability Boundary

These are community-maintained JLPT estimates, not official test specifications. The upstream project acknowledges that its original data came from `chyyran/jlpt-anki-decks`, based on decks from `tanos.co.uk`. Preserve this provenance and label assignments as `jlpt_estimate`.

The downloaded version has no empty expression, reading, or meaning fields. It contains two duplicate expression/reading keys inside N2 and 95 keys occurring in more than one level file. Catalogue generation must merge exact expression/reading duplicates, preserve all observed source levels, and choose one primary estimate through a documented deterministic rule rather than silently duplicating entries.
