#!/usr/bin/env python3
"""
Phase 1.4 — PDF Content Extraction

Given a registered PDF, reads its content map (from phase_1_3) and extracts
each page using the appropriate strategy:
- Plain text → extract via PyMuPDF
- Formula     → flag for manual review, extract text as-is
- Table       → prefer pdfplumber, fallback to PyMuPDF
- Image       → render full page as PNG to data/images/

Writes all text chunks to data/chunks.jsonl (one JSON per page).
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import fitz  # PyMuPDF

# pdfplumber is optional — provides better table extraction
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


REGISTRY_PATH = Path("data") / "registry.jsonl"
CONTENT_MAP_PATH = Path("data") / "content_map.jsonl"
CHUNKS_PATH = Path("data") / "chunks.jsonl"
IMAGES_DIR = Path("data") / "images"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    records: list[dict] = []
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return records


def _find_document_id(sha256: str, registry: list[dict]) -> str | None:
    for r in registry:
        if r.get("sha256") == sha256:
            return r["document_id"]
    return None


def _load_content_map(doc_id: str) -> list[dict]:
    """Load only the pages belonging to *doc_id* from the content map."""
    if not CONTENT_MAP_PATH.exists():
        return []
    pages: list[dict] = []
    with open(CONTENT_MAP_PATH, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if record.get("doc_id") == doc_id:
                    pages.append(record)
            except json.JSONDecodeError:
                continue
    return sorted(pages, key=lambda p: p["page"])


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------


def _get_ocr_lang(doc_id: str) -> str:
    """Determine Tesseract language string from the registry.

    Registry stores lang="kz" or "ru" (set in phase_1_1).
    Returns "kaz+rus" for Kazakh docs, "rus+kaz" otherwise.
    """
    registry = _load_registry()
    for r in registry:
        if r.get("document_id") == doc_id:
            lang = r.get("language", "ru")
            return "kaz+rus" if lang == "kz" else "rus+kaz"
    return "rus+kaz"


def _ocr_extract(pdf_path: str, page_num: int, doc_id: str) -> str:
    """Render page as image at 2× zoom and run OCR via pytesseract."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        from PIL import Image
        import pytesseract, io

        img = Image.open(io.BytesIO(img_bytes))
        ocr_lang = _get_ocr_lang(doc_id)
        ocr_text = pytesseract.image_to_string(
            img, lang=ocr_lang, config="--psm 1"
        )
        return ocr_text.strip()
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Extraction strategies
# ---------------------------------------------------------------------------


def _extract_text_via_fitz(pdf_path: str, page_num: int, doc_id: str) -> tuple[str, str]:
    """Extract plain text via PyMuPDF, falling back to OCR if < 300 chars.

    Returns (text, method) where method is "fitz" or "ocr".
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]  # fitz is 0-based
        fitz_text = page.get_text().strip()
        if len(fitz_text) < 300:
            print(f"  📷 Page {page_num}: fitz returned {len(fitz_text)} chars → OCR")
            ocr_text = _ocr_extract(pdf_path, page_num, doc_id)
            return ocr_text, "ocr"
        return fitz_text, "fitz"
    finally:
        doc.close()


def _extract_table_text(pdf_path: str, page_num: int, doc_id: str) -> tuple[str, str]:
    """
    Extract page text, preferring pdfplumber for table-aware extraction.

    Falls back to fitz OCR chain when pdfplumber is unavailable or fails.
    Returns (text, method).
    """
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                plumber_page = pdf.pages[page_num - 1]
                tables = plumber_page.find_tables()
                if tables:
                    # Render detected tables as tab-separated rows
                    lines: list[str] = []
                    for table in tables:
                        for row in table.extract():
                            cells = [
                                str(cell).strip() if cell else ""
                                for cell in row
                            ]
                            lines.append("\t".join(cells))
                    return "\n".join(lines), "pdfplumber"
        except Exception:
            logger.warning("⚠  pdfplumber failed, falling back to fitz")

    return _extract_text_via_fitz(pdf_path, page_num, doc_id)


def _save_page_image(pdf_path: str, page_num: int, doc_id: str) -> str | None:
    """Render the page as a PNG and save to data/images/."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]
        pix = page.get_pixmap()
        image_filename = f"{doc_id}_p{page_num}.png"
        image_path = IMAGES_DIR / image_filename
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        pix.save(str(image_path))
        return str(image_path)
    except Exception:
        return None
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Extraction orchestration
# ---------------------------------------------------------------------------


def extract(pdf_path: str) -> list[dict]:
    """Orchestrate per-page extraction using the content map."""
    path = Path(pdf_path).resolve()

    if not path.exists():
        print(f"❌  Error: File not found — {path}")
        raise SystemExit(1)

    if path.suffix.lower() != ".pdf":
        print(f"❌  Error: Not a PDF file — {path.suffix}")
        raise SystemExit(1)

    # Resolve document_id from registry
    sha256 = _compute_sha256(path)
    registry = _load_registry()
    doc_id = _find_document_id(sha256, registry)

    if doc_id is None:
        print(
            "❌  Error: PDF not found in registry. "
            "Run phase_1_1_register.py first."
        )
        raise SystemExit(1)

    # Load per-page content metadata
    content_pages = _load_content_map(doc_id)
    if not content_pages:
        print(
            "❌  Error: No content map found for this document. "
            "Run phase_1_3_detect_content.py first."
        )
        raise SystemExit(1)

    chunks: list[dict] = []
    for cp in content_pages:
        page_num = cp["page"]

        # --- Extract text -------------------------------------------------
        if cp.get("has_table"):
            text, extraction_method = _extract_table_text(str(path), page_num, doc_id)
        else:
            text, extraction_method = _extract_text_via_fitz(str(path), page_num, doc_id)

        # --- Formula flagging ---------------------------------------------
        if cp.get("has_formula"):
            logger.warning("⚠  Formula page %d: manual review needed", page_num)

        # --- Image extraction ---------------------------------------------
        image_path: str | None = None
        if cp.get("has_image"):
            image_path = _save_page_image(str(path), page_num, doc_id)

        chunk = {
            "doc_id": doc_id,
            "page": page_num,
            "text": text,
            "extraction_method": extraction_method,
            "formula_present": cp.get("has_formula", False),
            "table_present": cp.get("has_table", False),
            "has_image": cp.get("has_image", False),
            "image_path": image_path,
        }
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.4 — Extract text/chunks from a PDF using the content map."
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    chunks = extract(args.pdf_path)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    total_images = sum(1 for c in chunks if c["has_image"] and c["image_path"])
    total_formula = sum(1 for c in chunks if c["formula_present"])
    total_table = sum(1 for c in chunks if c["table_present"])

    print(
        f"✅  Extracted: {chunks[0]['doc_id']}  |  "
        f"{len(chunks)} chunks  |  "
        f"{total_formula} formula  |  "
        f"{total_table} table  |  "
        f"{total_images} images saved"
    )


if __name__ == "__main__":
    main()
