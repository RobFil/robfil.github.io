#!/usr/bin/env python
"""Track, extract, review, and publish Japanese learning sources.

Extraction and index management are deterministic. A reviewing agent must
compare the generated page images with the extracted text before accepting a
source into the context files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from parse_pdfs import (
    PageText,
    apply_ocr,
    default_ocr_language,
    default_tessdata_dir,
    extract_pdf,
    iter_pdfs,
    select_language_profile,
)


INDEX_VERSION = 1


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id(source_file: str) -> str:
    return hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:16]


def infer_source_metadata(source_file: str) -> dict[str, str | None]:
    """Infer date, thematic group, and document role conservatively from a filename."""
    stem = Path(source_file).stem
    title = stem
    document_date: str | None = None
    dated = re.match(r"^(\d{8})[_\-\s]+(.+)$", stem)
    if dated:
        try:
            document_date = datetime.strptime(dated.group(1), "%Y%m%d").date().isoformat()
            title = dated.group(2)
        except ValueError:
            pass

    parts = [part for part in re.split(r"[_\s]+", title) if part]
    role = "content"
    role_names = {
        "教材": "material",
        "解答": "answer",
        "回答": "answer",
        "問題": "exercise",
        "material": "material",
        "handout": "material",
        "answer": "answer",
        "answers": "answer",
        "solution": "answer",
        "solutions": "answer",
        "exercise": "exercise",
        "worksheet": "exercise",
    }
    while len(parts) > 1:
        suffix = parts[-1]
        normalized = suffix.casefold()
        if suffix == "すべて":
            parts.pop()
            continue
        if normalized in role_names:
            role = role_names[normalized]
            parts.pop()
            continue
        if suffix.isdigit():
            role = f"part-{suffix}"
            parts.pop()
            continue
        break

    topic_group = " ".join(parts).strip() or title
    return {
        "document_date": document_date,
        "topic_group": topic_group,
        "document_role": role,
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": INDEX_VERSION, "updated_at": None, "files": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != INDEX_VERSION or not isinstance(data.get("files"), dict):
        raise ValueError(f"Unsupported or invalid extraction index: {path}")
    return data


def save_index(path: Path, index: dict[str, Any]) -> None:
    index["updated_at"] = now_utc()
    atomic_write_text(path, json.dumps(index, ensure_ascii=False, indent=2) + "\n")


def relative_to_skill(path: Path) -> str:
    return path.resolve().relative_to(skill_root().resolve()).as_posix()


def artifact_paths(output_dir: Path, identifier: str) -> tuple[Path, Path, Path]:
    machine = output_dir / "machine" / "sources" / f"{identifier}.jsonl"
    human = output_dir / "human" / "review" / f"{identifier}.md"
    images = output_dir / "debug" / "review" / identifier
    return machine, human, images


def render_review_images(pdf_path: Path, target_dir: Path, dpi: int) -> int:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to render review images.") from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    for old_image in target_dir.glob("page-*.png"):
        old_image.unlink()
    scale = dpi / 72.0
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pixmap.save(target_dir / f"page-{index:04d}.png")
        return document.page_count


def build_records(
    pages: list[PageText],
    source_file: str,
    identifier: str,
    language_profile: str,
    language_hint: str,
    source_metadata: dict[str, str | None],
) -> list[dict[str, Any]]:
    classifier, detector = select_language_profile(language_profile)
    records: list[dict[str, Any]] = []
    for page in pages:
        if not page.text.strip():
            continue
        theme, confidence = classifier(page.text) if classifier else ("uncategorized", "low")
        detected_language = language_hint
        if language_hint == "auto":
            detected_language = detector(page.text) if detector else "unknown"
        records.append(
            {
                "id": f"{identifier}-p{page.page:04d}-0001",
                "source_file": source_file,
                "page": page.page,
                "document_date": source_metadata["document_date"],
                "topic_group": source_metadata["topic_group"],
                "document_role": source_metadata["document_role"],
                "theme": theme,
                "kind": "extracted_text",
                "text": page.text,
                "language_hint": detected_language,
                "extraction_method": page.extraction_method,
                "generated": False,
                "confidence": confidence,
                "review_status": "pending",
            }
        )
    return records


def write_source_artifacts(
    records: list[dict[str, Any]],
    pages: list[PageText],
    machine_path: Path,
    human_path: Path,
) -> None:
    jsonl = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    atomic_write_text(machine_path, jsonl)

    record_by_page = {record["page"]: record for record in records}
    blocks = ["# Extraktionsprüfung", ""]
    for page in pages:
        record = record_by_page.get(page.page)
        blocks.extend([f"## Page {page.page}", ""])
        if record:
            blocks.extend(
                [
                    f"- Theme: `{record['theme']}`",
                    f"- Language: `{record['language_hint']}`",
                    f"- Extraction: `{record['extraction_method']}`",
                    "",
                    record["text"],
                    "",
                ]
            )
        else:
            blocks.extend([f"TODO(review): {page.warning or 'No text extracted.'}", ""])
    atomic_write_text(human_path, "\n".join(blocks).rstrip() + "\n")


def markdown_cell(value: Any) -> str:
    """Render a value safely inside a Markdown table cell."""
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def extraction_label(entry: dict[str, Any]) -> str:
    extraction = entry.get("extraction") or {}
    ocr_pages = int(extraction.get("ocr_pages") or 0)
    if ocr_pages:
        return f"OCR ({ocr_pages} Seiten, {extraction.get('ocr_language') or 'Sprache unbekannt'})"
    return str(extraction.get("pdf_method") or "pdf_text")


def review_header(entry: dict[str, Any]) -> str:
    """Build the user-facing metadata header for one review artifact."""
    source = entry["source_file"]
    identifier = entry["source_id"]
    notes = entry.get("review_notes") or "Noch keine Reviewentscheidung."
    excluded_pages = entry.get("excluded_pages") or []
    excluded_label = ", ".join(str(page) for page in excluded_pages) or "keine"
    lines = [
        "# Extraktionsprüfung",
        "",
        f"- Quelldatei: `{source}`",
        f"- Quellen-ID: `{identifier}`",
        f"- Dokumentdatum: `{entry.get('document_date') or 'undatiert'}`",
        f"- Themengruppe: `{entry.get('topic_group') or 'nicht zugeordnet'}`",
        f"- Dokumentrolle: `{entry.get('document_role') or 'content'}`",
        f"- Extraktion: `{extraction_label(entry)}`",
        f"- Extraktionsversuche: `{entry.get('attempts', 0)}`",
        f"- Status: `{entry.get('status', 'unbekannt')}`",
        f"- Vom Kontext ausgeschlossene Seiten: `{excluded_label}`",
        f"- Ausschlussgrund: {entry.get('exclusion_reason') or '—'}",
        f"- Seitenbilder: [`output/debug/review/{identifier}/`](../../debug/review/{identifier}/)",
        f"- Maschinendaten: [`output/machine/sources/{identifier}.jsonl`](../../machine/sources/{identifier}.jsonl)",
        f"- Prüfhinweis: {notes}",
        "",
        "---",
        "",
    ]
    return "\n".join(lines) + "\n"


def refresh_review_outputs(index: dict[str, Any], output_dir: Path) -> int:
    """Refresh review headers and the central human-readable review index."""
    refreshed = 0
    entries = sorted(
        index["files"].values(),
        key=lambda entry: (
            entry.get("document_date") or "9999-12-31",
            entry["source_file"].casefold(),
        ),
    )
    for entry in entries:
        review_path_value = entry.get("review_markdown")
        if not review_path_value:
            continue
        review_path = skill_root() / review_path_value
        if not review_path.exists():
            continue
        existing = review_path.read_text(encoding="utf-8")
        page_start = re.search(r"(?m)^## Page \d+\s*$", existing)
        body = existing[page_start.start():] if page_start else ""
        atomic_write_text(review_path, review_header(entry) + body.rstrip() + "\n")
        refreshed += 1

    rows = [
        "# Review-Index",
        "",
        "> Zuordnung der menschenlesbaren Prüfdateien zu den ursprünglichen PDFs. "
        "Die Quellen-ID verbindet Review-MD, JSONL und Seitenbilder; der SHA-256-Inhaltshash "
        "für die Änderungs- und Duplikaterkennung steht im maschinellen Index.",
        "",
        "| Datum | Quelldatei | Quellen-ID | Gruppe / Rolle | Extraktion | Versuche | Ausgeschlossen | Status | Artefakte |",
        "|---|---|---|---|---|---:|---|---|---|",
    ]
    for entry in entries:
        identifier = entry.get("source_id") or "—"
        review_link = f"[Review](review/{identifier}.md)" if entry.get("review_markdown") else "—"
        image_link = f"[Seitenbilder](../debug/review/{identifier}/)" if entry.get("review_images") else "—"
        rows.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(entry.get("document_date") or "undatiert"),
                    markdown_cell(entry["source_file"]),
                    f"`{markdown_cell(identifier)}`",
                    markdown_cell(
                        f"{entry.get('topic_group') or 'nicht zugeordnet'} / "
                        f"{entry.get('document_role') or 'content'}"
                    ),
                    markdown_cell(extraction_label(entry)),
                    markdown_cell(entry.get("attempts", 0)),
                    markdown_cell(", ".join(str(page) for page in entry.get("excluded_pages") or []) or "—"),
                    f"`{markdown_cell(entry.get('status', 'unbekannt'))}`",
                    f"{review_link} · {image_link}",
                ]
            )
            + " |"
        )
    atomic_write_text(output_dir / "human" / "review-index.md", "\n".join(rows) + "\n")
    return refreshed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def ordered_topic_groups(
    entries: Iterable[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        inferred = infer_source_metadata(entry["source_file"])
        entry.setdefault("document_date", inferred["document_date"])
        entry.setdefault("topic_group", inferred["topic_group"])
        entry.setdefault("document_role", inferred["document_role"])
        grouped.setdefault(str(entry["topic_group"]), []).append(entry)

    def entry_sort_key(entry: dict[str, Any]) -> tuple[str, str]:
        return (entry.get("document_date") or "9999-12-31", entry["source_file"].casefold())

    return sorted(
        grouped.items(),
        key=lambda item: (
            min(entry_sort_key(entry)[0] for entry in item[1]),
            item[0].casefold(),
        ),
    )


def rebuild_context(index: dict[str, Any], output_dir: Path) -> None:
    accepted = [
        entry
        for entry in index["files"].values()
        if entry.get("status") == "accepted"
    ]
    ordered_groups = ordered_topic_groups(accepted)

    def entry_sort_key(entry: dict[str, Any]) -> tuple[str, str]:
        return (entry.get("document_date") or "9999-12-31", entry["source_file"].casefold())

    all_records: list[dict[str, Any]] = []
    markdown = [
        "# Reviewed Japanese Learning Context",
        "",
        "> Only source-faithful extractions accepted after visual and linguistic review.",
        "",
    ]
    for topic_group, entries in ordered_groups:
        markdown.extend([f"## Topic: {topic_group}", ""])
        for entry in sorted(entries, key=entry_sort_key):
            artifact = skill_root() / entry["artifact_jsonl"]
            excluded_pages = {int(page) for page in entry.get("excluded_pages") or []}
            records = [
                record
                for record in read_jsonl(artifact)
                if record.get("page") not in excluded_pages
            ]
            for record in records:
                record["review_status"] = "accepted"
                record["reviewed_at"] = entry.get("accepted_at")
                record["document_date"] = entry.get("document_date")
                record["topic_group"] = topic_group
                record["document_role"] = entry.get("document_role", "content")
            all_records.extend(records)

            date_label = entry.get("document_date") or "undated"
            markdown.extend(
                [
                    f"### {date_label} — {entry['source_file']}",
                    "",
                    f"- Role: `{entry.get('document_role', 'content')}`",
                    f"- SHA-256: `{entry['sha256']}`",
                    f"- Accepted: `{entry.get('accepted_at')}`",
                    f"- Review: {entry.get('review_notes', '')}",
                    f"- Excluded pages: `{', '.join(str(page) for page in sorted(excluded_pages)) or 'none'}`",
                    f"- Exclusion reason: {entry.get('exclusion_reason') or '—'}",
                    "",
                ]
            )
            for record in records:
                markdown.extend(
                    [
                        f"#### Page {record['page']} — {record['theme']}",
                        "",
                        record["text"],
                        "",
                    ]
                )

    context_jsonl = output_dir / "machine" / "context.jsonl"
    context_md = output_dir / "human" / "context.md"
    atomic_write_text(
        context_jsonl,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in all_records),
    )
    atomic_write_text(context_md, "\n".join(markdown).rstrip() + "\n")


def resolve_entry(index: dict[str, Any], selector: str) -> tuple[str, dict[str, Any]]:
    files = index["files"]
    if selector in files:
        return selector, files[selector]
    matches = [
        (name, entry)
        for name, entry in files.items()
        if entry.get("source_id") == selector or Path(name).name == selector
    ]
    if len(matches) != 1:
        raise ValueError(f"Source selector is missing or ambiguous: {selector}")
    return matches[0]


def selected_pdfs(input_dir: Path, filenames: list[str] | None) -> Iterable[Path]:
    pdfs = list(iter_pdfs(input_dir))
    if not filenames:
        return pdfs
    requested = set(filenames)
    return [pdf for pdf in pdfs if pdf.name in requested]


def command_status(args: argparse.Namespace) -> int:
    index = load_index(args.index)
    current = {pdf.name: pdf for pdf in iter_pdfs(args.input_dir)}
    names = sorted(set(index["files"]) | set(current), key=str.casefold)
    counts: Counter[str] = Counter()
    for name in names:
        entry = index["files"].get(name)
        metadata = entry or infer_source_metadata(name)
        if entry is None:
            status = "new"
        elif name not in current:
            status = "source_missing"
        elif sha256_file(current[name]) != entry.get("sha256"):
            status = "changed"
        else:
            status = entry.get("status", "unknown")
        counts[status] += 1
        date_label = metadata.get("document_date") or "undated"
        group_label = metadata.get("topic_group") or infer_source_metadata(name)["topic_group"]
        print(f"{status:20} {date_label} [{group_label}] {name}")
    print("Summary:", ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))
    return 0


def command_extract_new(args: argparse.Namespace) -> int:
    index = load_index(args.index)
    input_pdfs = list(selected_pdfs(args.input_dir, args.filename))
    present_names = {pdf.name for pdf in iter_pdfs(args.input_dir)}
    changed_index = False

    for name, entry in index["files"].items():
        if name not in present_names and entry.get("status") != "source_missing":
            entry["status"] = "source_missing"
            entry["status_changed_at"] = now_utc()
            changed_index = True

    extracted = 0
    skipped = 0
    for pdf_path in input_pdfs:
        digest = sha256_file(pdf_path)
        existing = index["files"].get(pdf_path.name)
        eligible = (
            args.force
            or existing is None
            or existing.get("sha256") != digest
            or existing.get("status") in {"needs_retry", "extraction_failed", "source_missing"}
        )
        if not eligible:
            skipped += 1
            continue
        if args.limit is not None and extracted >= args.limit:
            break

        identifier = source_id(pdf_path.name)
        source_metadata = infer_source_metadata(pdf_path.name)
        machine_path, human_path, image_dir = artifact_paths(args.output_dir, identifier)
        pages, pdf_method = extract_pdf(pdf_path)
        ocr_language = args.ocr_language or default_ocr_language(
            args.language_profile, args.language_hint
        )
        ocr_count = apply_ocr(
            pdf_path,
            pages,
            mode=args.ocr,
            language=ocr_language,
            dpi=args.ocr_dpi,
            page_segmentation_mode=args.ocr_psm,
            tesseract_command=args.tesseract_cmd,
            tessdata_dir=args.tessdata_dir or default_tessdata_dir(),
        )
        records = build_records(
            pages,
            pdf_path.name,
            identifier,
            args.language_profile,
            args.language_hint,
            source_metadata,
        )
        write_source_artifacts(records, pages, machine_path, human_path)
        rendered_pages = render_review_images(pdf_path, image_dir, args.review_dpi)

        attempts = int(existing.get("attempts", 0)) + 1 if existing else 1
        history = existing.get("review_history", []) if existing else []
        status = "pending_review" if records else "extraction_failed"
        index["files"][pdf_path.name] = {
            "source_file": pdf_path.name,
            "source_id": identifier,
            "document_date": source_metadata["document_date"],
            "topic_group": source_metadata["topic_group"],
            "document_role": source_metadata["document_role"],
            "sha256": digest,
            "size_bytes": pdf_path.stat().st_size,
            "modified_utc": datetime.fromtimestamp(
                pdf_path.stat().st_mtime, timezone.utc
            ).isoformat(timespec="seconds"),
            "status": status,
            "attempts": attempts,
            "parsed_at": now_utc(),
            "accepted_at": None,
            "page_count": len(pages),
            "rendered_review_pages": rendered_pages,
            "record_count": len(records),
            "artifact_jsonl": relative_to_skill(machine_path),
            "review_markdown": relative_to_skill(human_path),
            "review_images": relative_to_skill(image_dir),
            "extraction": {
                "pdf_method": pdf_method,
                "ocr_mode": args.ocr,
                "ocr_pages": ocr_count,
                "ocr_language": ocr_language,
                "ocr_dpi": args.ocr_dpi,
                "ocr_psm": args.ocr_psm,
            },
            "review_history": history,
            "review_notes": None,
            "excluded_pages": [],
            "exclusion_reason": None,
        }
        extracted += 1
        changed_index = True
        print(f"{status:20} {pdf_path.name} ({len(records)} records, OCR pages={ocr_count})")

    if changed_index:
        save_index(args.index, index)
        refresh_review_outputs(index, args.output_dir)
        rebuild_context(index, args.output_dir)
    print(f"Extracted={extracted}, skipped={skipped}")
    return 0


def command_review(args: argparse.Namespace) -> int:
    index = load_index(args.index)
    name, entry = resolve_entry(index, args.source)
    source_path = args.input_dir / name
    if not source_path.is_file():
        raise ValueError(f"Original source is missing: {source_path}")
    if sha256_file(source_path) != entry.get("sha256"):
        raise ValueError("Source changed after extraction; run extract-new before review.")

    if args.document_date:
        try:
            entry["document_date"] = datetime.strptime(
                args.document_date, "%Y-%m-%d"
            ).date().isoformat()
        except ValueError as exc:
            raise ValueError("--document-date must use YYYY-MM-DD.") from exc
    if args.topic_group:
        entry["topic_group"] = args.topic_group.strip()
    if args.document_role:
        entry["document_role"] = args.document_role.strip()

    excluded_pages = sorted(set(args.exclude_page or []))
    if excluded_pages and args.decision != "accept":
        raise ValueError("--exclude-page can only be used with --decision accept.")
    if excluded_pages and not args.exclusion_reason:
        raise ValueError("--exclusion-reason is required with --exclude-page.")
    invalid_pages = [
        page for page in excluded_pages if page < 1 or page > int(entry.get("page_count", 0))
    ]
    if invalid_pages:
        raise ValueError(f"Excluded page numbers are outside the PDF: {invalid_pages}")

    if args.decision == "accept":
        if entry.get("status") != "pending_review":
            raise ValueError("Only a pending_review extraction can be accepted.")
        artifact = skill_root() / entry["artifact_jsonl"]
        remaining_records = [
            record
            for record in read_jsonl(artifact)
            if record.get("page") not in excluded_pages
        ]
        if not remaining_records:
            raise ValueError("Cannot accept an extraction without records.")
        entry["status"] = "accepted"
        entry["accepted_at"] = now_utc()
        entry["excluded_pages"] = excluded_pages
        entry["exclusion_reason"] = args.exclusion_reason.strip() if excluded_pages else None
    elif args.decision == "retry":
        entry["status"] = "needs_retry"
        entry["accepted_at"] = None
    else:
        entry["status"] = "needs_manual_review"
        entry["accepted_at"] = None

    entry["review_notes"] = args.notes
    entry.setdefault("review_history", []).append(
        {
            "reviewed_at": now_utc(),
            "decision": args.decision,
            "attempt": entry.get("attempts"),
            "notes": args.notes,
            "excluded_pages": excluded_pages,
            "exclusion_reason": args.exclusion_reason,
        }
    )
    save_index(args.index, index)
    refresh_review_outputs(index, args.output_dir)
    rebuild_context(index, args.output_dir)
    print(f"{entry['status']:20} {name}")
    return 0


def command_refresh_review(args: argparse.Namespace) -> int:
    index = load_index(args.index)
    migrated = False
    for entry in index["files"].values():
        if "excluded_pages" not in entry:
            entry["excluded_pages"] = []
            migrated = True
        if "exclusion_reason" not in entry:
            entry["exclusion_reason"] = None
            migrated = True
    if migrated:
        save_index(args.index, index)
    refreshed = refresh_review_outputs(index, args.output_dir)
    print(
        f"Refreshed review files={refreshed}; "
        f"migrated index entries={len(index['files']) if migrated else 0}; "
        "index=output/human/review-index.md"
    )
    return 0


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    root = skill_root()
    parser.add_argument("--input-dir", type=Path, default=root / "input" / "pdfs")
    parser.add_argument("--output-dir", type=Path, default=root / "output")
    parser.add_argument(
        "--index", type=Path, default=root / "output" / "machine" / "parsed_files_index.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage indexed PDF extraction and review.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="List new, changed, and reviewed files.")
    add_common_paths(status)
    status.set_defaults(function=command_status)

    refresh = subparsers.add_parser(
        "refresh-review", help="Refresh review Markdown headers and the review index."
    )
    add_common_paths(refresh)
    refresh.set_defaults(function=command_refresh_review)

    extract = subparsers.add_parser(
        "extract-new", help="Extract only new, changed, or retry-requested PDFs."
    )
    add_common_paths(extract)
    extract.add_argument("--filename", action="append")
    extract.add_argument("--force", action="store_true")
    extract.add_argument("--limit", type=int)
    extract.add_argument("--language-profile", choices=("none", "japanese"), default="japanese")
    extract.add_argument(
        "--language-hint", choices=("auto", "ja", "de", "en", "mixed", "unknown"), default="auto"
    )
    extract.add_argument("--ocr", choices=("off", "missing", "all"), default="missing")
    extract.add_argument("--ocr-language")
    extract.add_argument("--ocr-dpi", type=int, default=300)
    extract.add_argument("--ocr-psm", type=int, default=3)
    extract.add_argument("--tesseract-cmd", type=Path)
    extract.add_argument("--tessdata-dir", type=Path)
    extract.add_argument("--review-dpi", type=int, default=150)
    extract.set_defaults(function=command_extract_new)

    review = subparsers.add_parser("review", help="Record the AI review decision.")
    add_common_paths(review)
    review.add_argument("source", help="Exact filename or source id.")
    review.add_argument("--decision", choices=("accept", "retry", "manual"), required=True)
    review.add_argument("--notes", required=True)
    review.add_argument("--document-date")
    review.add_argument("--topic-group")
    review.add_argument("--document-role")
    review.add_argument(
        "--exclude-page",
        action="append",
        type=int,
        help="Exclude one page from accepted context; repeat for multiple pages.",
    )
    review.add_argument(
        "--exclusion-reason",
        help="Required reason when one or more pages are excluded.",
    )
    review.set_defaults(function=command_review)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "ocr_dpi") and args.ocr_dpi < 72:
        parser.error("--ocr-dpi must be at least 72")
    if hasattr(args, "review_dpi") and args.review_dpi < 72:
        parser.error("--review-dpi must be at least 72")
    try:
        return args.function(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
