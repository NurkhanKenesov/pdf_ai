#!/usr/bin/env python3
"""
Phase 1.3 — Page-Level Content Detection

Scans each page of a PDF and detects the presence of four content types:
- text    — extractable text layer with >100 chars
- formula — math expressions (equals signs, Unicode operators, Greek letters)
- table   — grid-aligned text blocks or keyword match
- image   — embedded images via PyMuPDF

Outputs one JSONL record per page to data/content_map.jsonl.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz  # PyMuPDF


REGISTRY_PATH = Path("data") / "registry.jsonl"
CONTENT_MAP_PATH = Path("data") / "content_map.jsonl"

# Math patterns commonly found in Kazakh/Russian ENT textbooks
MATH_PATTERN = re.compile(
    r"(?<![<>=!:;])="           # standalone equals (not ==, <=, >=, !=)
    r"|[∫²³√±×÷∑∏∞∂∆≈≠≤≥]"    # Unicode math operators
    r"|[α-ωΑ-Ω]"                # Greek letters used in formulae
    r"|\d+\s*[+\-−×÷]\s*\d+"   # digit-operator-digit (e.g. 3+5, 2x−1)
)

TABLE_KEYWORDS = {"кесте", "таблица", "table"}


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


# ---------------------------------------------------------------------------
# Per-page detectors
# ---------------------------------------------------------------------------


def _has_text(page) -> bool:
    """Page has at least 100 characters of extractable text."""
    text = page.get_text().strip()
    return len(text) > 100


def _has_formula(page) -> bool:
    """Detect math expressions via pattern matching (>3 hits = formula page)."""
    text = page.get_text()
    return len(MATH_PATTERN.findall(text)) > 3


def _has_table(page) -> bool:
    """Detect tables via grid-aligned blocks or keyword match."""
    text = page.get_text().lower()

    # Keyword shortcut
    if any(kw in text for kw in TABLE_KEYWORDS):
        return True

    # Grid-structure check via fitz text-block positions
    blocks = page.get_text("dict").get("blocks", [])
    text_blocks = [b for b in blocks if b.get("type") == 0]

    if len(text_blocks) < 4:
        return False

    # Group by x0 rounded to nearest 30 px column
    col_groups: dict[int, list] = {}
    for block in text_blocks:
        x0 = block["bbox"][0]
        col = round(x0 / 30)
        col_groups.setdefault(col, []).append(block)

    # If any column has more than 3 rows, likely a table
    return any(len(blocks_in_col) > 3 for blocks_in_col in col_groups.values())


def _has_image(page) -> bool:
    """Check for embedded images on the page."""
    return len(page.get_images()) > 0


# ---------------------------------------------------------------------------
# Content detection
# ---------------------------------------------------------------------------


def detect_content(pdf_path: str) -> list[dict]:
    """Analyse every page and return a list of content-map records."""
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

    doc = fitz.open(str(path))
    try:
        total_pages = doc.page_count
        page_records: list[dict] = []

        for i in range(total_pages):
            page = doc[i]
            record = {
                "doc_id": doc_id,
                "page": i + 1,  # 1-based for human readability
                "has_text": _has_text(page),
                "has_formula": _has_formula(page),
                "has_table": _has_table(page),
                "has_image": _has_image(page),
            }
            page_records.append(record)
    finally:
        doc.close()

    return page_records


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.3 — Detect per-page content (text, formula, table, image)."
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    records = detect_content(args.pdf_path)

    CONTENT_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONTENT_MAP_PATH, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with_formula = sum(1 for r in records if r["has_formula"])
    with_table = sum(1 for r in records if r["has_table"])
    with_image = sum(1 for r in records if r["has_image"])
    print(
        f"✅  Content mapped: {records[0]['doc_id']}  |  "
        f"{len(records)} pages  |  "
        f"{with_formula} formula  |  "
        f"{with_table} table  |  "
        f"{with_image} image"
    )


if __name__ == "__main__":
    main()
