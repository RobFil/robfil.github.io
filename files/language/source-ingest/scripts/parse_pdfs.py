#!/usr/bin/env python
"""AI-free, best-effort text extraction from PDF files.

The script intentionally avoids hard dependency installation. It uses pdfplumber
when available, falls back to pypdf, and can OCR image-only pages locally.
OCR and language-specific classification are kept in separate modules.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ThemeClassifier = Callable[[str], tuple[str, str]]
LanguageDetector = Callable[[str], str]


@dataclass
class PageText:
    source_file: str
    page: int
    text: str
    warning: str | None = None
    extraction_method: str = "pdf_text"


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or "source"


def select_language_profile(
    profile: str,
) -> tuple[ThemeClassifier | None, LanguageDetector | None]:
    if profile == "japanese":
        from japanese_support import classify_theme as classify_japanese_theme
        from japanese_support import detect_language_hint as detect_japanese_language_hint

        return classify_japanese_theme, detect_japanese_language_hint
    return None, None


def extract_with_pdfplumber(pdf_path: Path) -> list[PageText]:
    import pdfplumber  # type: ignore

    pages: list[PageText] = []
    with pdfplumber.open(pdf_path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            warning = None if text.strip() else "No extractable text; page may require OCR."
            pages.append(PageText(pdf_path.name, index, text.strip(), warning))
    return pages


def extract_with_pypdf(pdf_path: Path) -> list[PageText]:
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        warning = None if text.strip() else "No extractable text; page may require OCR."
        pages.append(PageText(pdf_path.name, index, text.strip(), warning))
    return pages


def extract_pdf(pdf_path: Path) -> tuple[list[PageText], str]:
    try:
        return extract_with_pdfplumber(pdf_path), "pdfplumber"
    except ImportError:
        pass
    except Exception as exc:
        return [PageText(pdf_path.name, 0, "", f"pdfplumber failed: {exc}")], "pdfplumber"

    try:
        return extract_with_pypdf(pdf_path), "pypdf"
    except ImportError:
        message = "Install pdfplumber or pypdf to extract PDF text."
        return [PageText(pdf_path.name, 0, "", message)], "none"
    except Exception as exc:
        return [PageText(pdf_path.name, 0, "", f"pypdf failed: {exc}")], "pypdf"


def iter_pdfs(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.glob("*.pdf"))


def default_ocr_language(language_profile: str, language_hint: str) -> str:
    if language_profile == "japanese":
        return "jpn+deu+eng"
    return {"ja": "jpn", "de": "deu", "en": "eng"}.get(language_hint, "eng")


def default_tessdata_dir() -> Path | None:
    candidate = skill_root().parents[1] / ".venv" / "tools" / "tessdata"
    return candidate if candidate.is_dir() else None


def apply_ocr(
    pdf_path: Path,
    pages: list[PageText],
    mode: str,
    language: str,
    dpi: int,
    page_segmentation_mode: int,
    tesseract_command: Path | None,
    tessdata_dir: Path | None,
) -> int:
    """OCR missing or all pages in place and return the successful page count."""
    if mode == "off":
        return 0

    targets = [
        page.page
        for page in pages
        if page.page > 0 and (mode == "all" or not page.text.strip())
    ]
    if not targets:
        return 0

    try:
        from ocr_support import OCRUnavailable, ocr_pdf_pages

        recognized = ocr_pdf_pages(
            pdf_path,
            targets,
            language=language,
            dpi=dpi,
            page_segmentation_mode=page_segmentation_mode,
            tesseract_command=tesseract_command,
            tessdata_dir=tessdata_dir,
        )
    except (OCRUnavailable, OSError) as exc:
        for page in pages:
            if page.page in targets and not page.text.strip():
                page.warning = f"OCR unavailable: {exc}"
        return 0

    count = 0
    for page in pages:
        if page.page not in targets:
            continue
        text = recognized.get(page.page, "").strip()
        if text:
            page.text = text
            page.warning = None
            page.extraction_method = "ocr"
            count += 1
        elif not page.text.strip():
            page.warning = f"OCR produced no text (language={language}, dpi={dpi})."
    return count


def write_outputs(
    pages: list[PageText],
    output_dir: Path,
    classifier: ThemeClassifier | None = None,
    language_detector: LanguageDetector | None = None,
    language_hint: str = "auto",
    language_profile: str = "none",
) -> int:
    machine_dir = output_dir / "machine"
    human_dir = output_dir / "human"
    debug_dir = output_dir / "debug"
    for directory in (machine_dir, human_dir, debug_dir):
        directory.mkdir(parents=True, exist_ok=True)

    machine_handles = {}
    human_blocks: dict[str, list[str]] = {}
    warnings: list[str] = []
    count = 0

    try:
        for page in pages:
            if page.warning:
                warnings.append(f"{page.source_file}, page {page.page}: {page.warning}")
            if not page.text:
                continue

            theme, confidence = (
                classifier(page.text) if classifier else ("uncategorized", "low")
            )
            record_language = language_hint
            if language_hint == "auto":
                record_language = language_detector(page.text) if language_detector else "unknown"
            source_stem = slugify(Path(page.source_file).stem)
            record_id = f"{source_stem}-p{page.page:04d}-0001"
            record = {
                "id": record_id,
                "source_file": page.source_file,
                "page": page.page,
                "theme": theme,
                "kind": "extracted_text",
                "text": page.text,
                "language_hint": record_language,
                "extraction_method": page.extraction_method,
                "generated": False,
                "confidence": confidence,
            }

            if theme not in machine_handles:
                machine_handles[theme] = (machine_dir / f"{theme}.jsonl").open("a", encoding="utf-8")
            machine_handles[theme].write(json.dumps(record, ensure_ascii=False) + "\n")

            human_blocks.setdefault(theme, []).append(
                f"## {page.source_file}, page {page.page}\n\n"
                f"Theme: {theme}\nConfidence: {confidence}\n\n{page.text}\n"
            )
            count += 1
    finally:
        for handle in machine_handles.values():
            handle.close()

    for theme, blocks in human_blocks.items():
        with (human_dir / f"{theme}.md").open("a", encoding="utf-8") as handle:
            handle.write("\n---\n\n".join(blocks))
            handle.write("\n")

    if warnings:
        with (debug_dir / "extraction_warnings.txt").open("a", encoding="utf-8") as handle:
            handle.write("\n".join(warnings) + "\n")

    manifest = {
        "records_written": count,
        "themes": sorted(human_blocks),
        "warnings": len(warnings),
        "language_profile": language_profile,
        "language_hint": language_hint,
    }
    with (machine_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return count


def main(argv: list[str] | None = None) -> int:
    # Avoid crashes when Japanese filenames are printed to a legacy Windows
    # console whose active encoding cannot represent them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    root = skill_root()
    parser = argparse.ArgumentParser(
        description="Extract PDF text without AI; optionally apply a language profile."
    )
    parser.add_argument("--input-dir", type=Path, default=root / "input" / "pdfs")
    parser.add_argument("--output-dir", type=Path, default=root / "output")
    parser.add_argument(
        "--language-profile",
        choices=("none", "japanese"),
        default="japanese",
        help="Optional deterministic rules for thematic classification (default: japanese).",
    )
    parser.add_argument(
        "--language-hint",
        choices=("auto", "ja", "de", "en", "mixed", "unknown"),
        default="auto",
        help="Language stored in output records; auto detection is profile-dependent.",
    )
    parser.add_argument(
        "--ocr",
        choices=("off", "missing", "all"),
        default="missing",
        help="OCR no pages, pages without text, or all pages (default: missing).",
    )
    parser.add_argument(
        "--ocr-language",
        help="Tesseract languages, e.g. jpn+deu+eng; derived from language options by default.",
    )
    parser.add_argument("--ocr-dpi", type=int, default=300)
    parser.add_argument("--ocr-psm", type=int, default=3)
    parser.add_argument("--tesseract-cmd", type=Path)
    parser.add_argument(
        "--tessdata-dir",
        type=Path,
        help="Directory containing Tesseract .traineddata files.",
    )
    args = parser.parse_args(argv)

    if args.ocr_dpi < 72:
        parser.error("--ocr-dpi must be at least 72")
    if not 0 <= args.ocr_psm <= 13:
        parser.error("--ocr-psm must be between 0 and 13")

    pdfs = list(iter_pdfs(args.input_dir))
    if not pdfs:
        print(f"No PDFs found in {args.input_dir}")
        return 0

    all_pages: list[PageText] = []
    methods: dict[str, str] = {}
    ocr_language = args.ocr_language or default_ocr_language(
        args.language_profile, args.language_hint
    )
    for pdf_path in pdfs:
        pages, method = extract_pdf(pdf_path)
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
        if ocr_count:
            method = f"{method}+ocr({ocr_count})"
        methods[pdf_path.name] = method
        all_pages.extend(pages)

    classifier, language_detector = select_language_profile(args.language_profile)
    count = write_outputs(
        all_pages,
        args.output_dir,
        classifier=classifier,
        language_detector=language_detector,
        language_hint=args.language_hint,
        language_profile=args.language_profile,
    )
    print(f"Processed {len(pdfs)} PDF(s), wrote {count} text record(s).")
    for filename, method in methods.items():
        print(f"{filename}: {method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
