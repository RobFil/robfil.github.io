"""Local, deterministic OCR support for PDF pages.

PyMuPDF renders pages to PNG in memory. Tesseract reads the PNG from stdin and
writes UTF-8 text to stdout. No cloud service or generative AI is used.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class OCRUnavailable(RuntimeError):
    """Raised when an OCR dependency or requested language is unavailable."""


def find_tesseract(explicit_command: Path | None = None) -> Path:
    """Locate Tesseract from an explicit path, PATH, or common Windows paths."""
    candidates: list[Path] = []
    if explicit_command:
        candidates.append(explicit_command.expanduser())

    path_command = shutil.which("tesseract")
    if path_command:
        candidates.append(Path(path_command))

    for environment_name in ("ProgramFiles", "LOCALAPPDATA"):
        base = os.environ.get(environment_name)
        if base:
            candidates.append(Path(base) / "Tesseract-OCR" / "tesseract.exe")
            candidates.append(Path(base) / "Programs" / "Tesseract-OCR" / "tesseract.exe")

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise OCRUnavailable(
        "Tesseract was not found. Install Tesseract OCR or pass --tesseract-cmd."
    )


def available_languages(
    tesseract_command: Path, tessdata_dir: Path | None = None
) -> set[str]:
    """Return installed Tesseract language identifiers."""
    command = [str(tesseract_command)]
    if tessdata_dir:
        command.extend(["--tessdata-dir", str(tessdata_dir)])
    command.append("--list-langs")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise OCRUnavailable(f"Tesseract language check failed: {detail}")
    return {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and not line.lower().startswith("list of available languages")
    }


def validate_languages(
    tesseract_command: Path, language: str, tessdata_dir: Path | None = None
) -> None:
    requested = {item.strip() for item in language.split("+") if item.strip()}
    installed = available_languages(tesseract_command, tessdata_dir)
    missing = sorted(requested - installed)
    if missing:
        raise OCRUnavailable(
            "Missing Tesseract language data: "
            f"{', '.join(missing)}. Installed: {', '.join(sorted(installed)) or 'none'}."
        )


def ocr_pdf_pages(
    pdf_path: Path,
    page_numbers: list[int],
    language: str,
    dpi: int = 300,
    page_segmentation_mode: int = 3,
    tesseract_command: Path | None = None,
    tessdata_dir: Path | None = None,
) -> dict[int, str]:
    """OCR selected 1-based PDF pages and return their recognized text."""
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise OCRUnavailable(
            "PyMuPDF is not installed. Install it with: python -m pip install pymupdf"
        ) from exc

    command = find_tesseract(tesseract_command)
    if tessdata_dir:
        tessdata_dir = tessdata_dir.expanduser().resolve()
        if not tessdata_dir.is_dir():
            raise OCRUnavailable(f"Tesseract data directory not found: {tessdata_dir}")
    validate_languages(command, language, tessdata_dir)
    requested_pages = sorted(set(page_numbers))
    results: dict[int, str] = {}

    with fitz.open(pdf_path) as document:
        for page_number in requested_pages:
            if page_number < 1 or page_number > document.page_count:
                raise OCRUnavailable(
                    f"Page {page_number} is outside PDF page range 1-{document.page_count}."
                )
            page = document.load_page(page_number - 1)
            scale = dpi / 72.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image_bytes = pixmap.tobytes("png")
            ocr_command = [str(command), "stdin", "stdout"]
            if tessdata_dir:
                ocr_command.extend(["--tessdata-dir", str(tessdata_dir)])
            ocr_command.extend(
                ["-l", language, "--psm", str(page_segmentation_mode)]
            )
            completed = subprocess.run(
                ocr_command,
                input=image_bytes,
                check=False,
                capture_output=True,
            )
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace").strip()
                raise OCRUnavailable(
                    f"Tesseract failed for page {page_number}: {detail or 'unknown error'}"
                )
            results[page_number] = completed.stdout.decode(
                "utf-8", errors="replace"
            ).strip()

    return results
