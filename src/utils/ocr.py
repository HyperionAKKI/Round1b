# src/utils/ocr.py
from pathlib import Path
from typing import List
import tempfile

from pdf2image import convert_from_path
import pytesseract


def ocr_pdf(pdf_path: str, dpi: int = 300) -> str:
    """
    Convert every page of `pdf_path` into an image and run Tesseract OCR.
    Returns the full text (pages separated by two new-lines).
    """
    pdf_path = Path(pdf_path)
    text_pages: List[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        images = convert_from_path(str(pdf_path), dpi=dpi, output_folder=tmp)
        for img in images:
            text_pages.append(pytesseract.image_to_string(img))

    return "\n\n".join(text_pages).strip()
