from collections import Counter
import math
import re
from pathlib import Path

import fitz
from docx import Document


def _normalize_extracted_text(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n+ *", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_repeated_page_margins(page_texts: list[str]) -> list[str]:
    if len(page_texts) < 2:
        return page_texts

    margin_counts = Counter()
    page_lines = []
    page_margin_indices = []
    for page_text in page_texts:
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        page_lines.append(lines)
        if len(lines) >= 5:
            margin_indices = set(range(2)) | set(range(len(lines) - 2, len(lines)))
        elif len(lines) >= 3:
            margin_indices = {0, len(lines) - 1}
        else:
            margin_indices = set(range(len(lines)))
        page_margin_indices.append(margin_indices)
        margin_lines = [lines[index] for index in margin_indices]
        for line in set(margin_lines):
            if len(line) <= 120:
                key = re.sub(r"\d+", "#", line.casefold())
                margin_counts[key] += 1

    repeated = {
        line for line, count in margin_counts.items()
        if count >= max(2, math.ceil(len(page_texts) * 0.6))
    }
    cleaned_pages = []
    for lines, margin_indices in zip(page_lines, page_margin_indices):
        kept = []
        for index, line in enumerate(lines):
            key = re.sub(r"\d+", "#", line.casefold())
            if index in margin_indices and len(line) <= 120 and key in repeated:
                continue
            kept.append(line)
        cleaned_pages.append("\n\n".join(kept))
    return cleaned_pages


def extract_pdf_text(file_obj, *, return_metadata=False):
    """Extract text from a PDF using PyMuPDF and return page and character metadata."""
    file_obj.seek(0)
    data = file_obj.read()

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # pragma: no cover - surfaced to UI as user-facing error
        raise ValueError(f"The PDF file is corrupted or invalid: {exc}") from exc

    try:
        if doc.is_encrypted:
            password_ok = doc.authenticate("")
            if not password_ok:
                raise ValueError("This PDF is password-protected. Please unlock it and upload again.")

        page_texts = []
        for page_number, page in enumerate(doc, start=1):
            try:
                page_text = page.get_text("text") or ""
            except Exception as exc:
                raise ValueError(f"Failed to read page {page_number}: {exc}") from exc
            cleaned = _normalize_extracted_text(page_text)
            page_texts.append(cleaned)

        page_texts = _remove_repeated_page_margins(page_texts)
        extracted_text = "\n\n".join(page_texts)
        metadata = {
            "pages": doc.page_count,
            "characters": len(extracted_text),
            "is_empty": not bool(extracted_text.strip()),
        }

        if not extracted_text.strip():
            raise ValueError(
                "No readable text was found in this PDF. It may be a scanned/image-only document; OCR is required."
            )

        return (extracted_text, metadata) if return_metadata else extracted_text
    finally:
        doc.close()


def extract_text(uploaded_file, *, return_metadata=False, filename=None):
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")

    suffix = Path(filename or getattr(uploaded_file, "name", "")).suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_text(uploaded_file, return_metadata=return_metadata)

    if suffix == ".docx":
        uploaded_file.seek(0)
        document = Document(uploaded_file)
        text = "\n".join(p.text for p in document.paragraphs)
        normalized = _normalize_extracted_text(text)
        metadata = {"pages": 1, "characters": len(normalized), "is_empty": not normalized.strip()}
        if not normalized.strip():
            raise ValueError("No readable text was found in this DOCX file.")
        return (normalized, metadata) if return_metadata else normalized

    if suffix == ".txt":
        uploaded_file.seek(0)
        raw = uploaded_file.getvalue()
        text = raw.decode("utf-8", errors="replace")
        normalized = _normalize_extracted_text(text)
        metadata = {"pages": 1, "characters": len(normalized), "is_empty": not normalized.strip()}
        if not normalized.strip():
            raise ValueError("The uploaded text file is empty.")
        return (normalized, metadata) if return_metadata else normalized

    raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
