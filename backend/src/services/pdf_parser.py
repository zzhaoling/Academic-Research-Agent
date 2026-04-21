import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any

def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a PDF file and extract text per page.
    Returns list of dicts: [{"page": int, "text": str}, ...]
    """
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i, "text": text})
    return pages