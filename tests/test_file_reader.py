from io import BytesIO

import fitz
import pytest

from utils.file_reader import extract_pdf_text, extract_text


def test_pdf_extraction_removes_repeated_headers_and_page_numbers():
    document = fitz.open()
    for page_number in range(1, 4):
        page = document.new_page()
        page.insert_text((50, 50), "Quarterly Report")
        page.insert_text((50, 100), f"Unique body content for page {page_number}.")
        page.insert_text((50, 780), f"Page {page_number}")

    pdf_bytes = document.tobytes()
    document.close()
    text, metadata = extract_pdf_text(BytesIO(pdf_bytes), return_metadata=True)

    assert "Quarterly Report" not in text
    assert "Page 1" not in text
    assert all(f"Unique body content for page {number}" in text for number in range(1, 4))
    assert metadata["pages"] == 3


def test_empty_pdf_reports_scanned_or_unreadable_document():
    document = fitz.open()
    document.new_page()
    pdf_bytes = document.tobytes()
    document.close()

    with pytest.raises(ValueError, match="scanned/image-only"):
        extract_pdf_text(BytesIO(pdf_bytes))


def test_extract_text_uses_supplied_filename_for_bytes_stream():
    text = extract_text(
        BytesIO(b"  Multilingual   text  "),
        filename="notes.txt",
    )

    assert text == "Multilingual text"
