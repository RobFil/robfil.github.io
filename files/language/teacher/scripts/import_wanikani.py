#!/usr/bin/env python
"""Download WaniKani vocabulary through the official API.

The API token is read only from WANIKANI_API_TOKEN. The resulting reference
cache is private local input and must not be committed or published.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


API_URL = "https://api.wanikani.com/v2/subjects?types=vocabulary,kana_vocabulary"
API_REVISION = "20170710"


def default_output() -> Path:
    return Path(__file__).resolve().parents[1] / "references" / "private" / "wanikani_vocabulary.jsonl"


def api_pages(url: str, token: str, timeout: float = 30.0) -> Iterator[dict[str, Any]]:
    while url:
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Wanikani-Revision": API_REVISION,
                "User-Agent": "local-japanese-teacher-reference-import/1.0",
            },
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    page = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 3:
                    time.sleep(min(float(exc.headers.get("Retry-After", "1")), 30.0))
                    continue
                if exc.code in {401, 403}:
                    raise RuntimeError("WaniKani rejected the API token or its authorization.") from exc
                raise RuntimeError(f"WaniKani API returned HTTP {exc.code}.") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Could not reach the WaniKani API: {exc.reason}") from exc
        yield page
        url = page.get("pages", {}).get("next_url") or ""


def subject_record(subject: dict[str, Any]) -> dict[str, Any] | None:
    if subject.get("object") not in {"vocabulary", "kana_vocabulary"}:
        return None
    data = subject.get("data", {})
    characters = data.get("characters")
    if not characters:
        return None

    readings = [
        item["reading"]
        for item in data.get("readings", [])
        if item.get("accepted_answer") and item.get("reading")
    ]
    primary_meanings = [
        item["meaning"]
        for item in data.get("meanings", [])
        if item.get("accepted_answer") and item.get("meaning")
    ]
    if not primary_meanings:
        primary_meanings = [
            item["meaning"] for item in data.get("meanings", []) if item.get("meaning")
        ]

    source_version = subject.get("data_updated_at") or data.get("created_at") or "unknown"
    return {
        "id": f"wanikani-{subject['id']}",
        "lemma": characters,
        "reading": readings[0] if readings else None,
        "readings": readings,
        "part_of_speech": data.get("parts_of_speech", []),
        "translations": {
            "en": [
                {
                    "text": meaning,
                    "source": f"wanikani:{source_version}",
                    "generated": False,
                    "review_status": "accepted",
                }
                for meaning in primary_meanings
            ]
        },
        "wanikani_level": data.get("level"),
        "wanikani_subject_id": subject["id"],
        "wanikani_object": subject.get("object"),
        "source_url": data.get("document_url"),
        "source_updated_at": subject.get("data_updated_at"),
        "hidden_at": data.get("hidden_at"),
    }


def write_records(pages: Iterator[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for page in pages:
            for subject in page.get("data", []):
                record = subject_record(subject)
                if record is None:
                    continue
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    temporary.replace(output)

    metadata = {
        "source": "WaniKani API v2 subjects",
        "api_revision": API_REVISION,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record_count": count,
        "private_local_reference": True,
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Transform one saved API collection without network access (for testing).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input_json:
        page = json.loads(args.input_json.read_text(encoding="utf-8"))
        pages = iter([page])
    else:
        token = os.environ.get("WANIKANI_API_TOKEN", "").strip()
        if not token:
            print("WANIKANI_API_TOKEN is not set.", file=sys.stderr)
            return 2
        pages = api_pages(API_URL, token, args.timeout)

    try:
        count = write_records(pages, args.output.resolve())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1
    print(f"Imported {count} WaniKani vocabulary records to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
