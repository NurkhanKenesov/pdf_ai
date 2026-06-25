#!/usr/bin/env python3
"""
Phase 1.2 — PDF Type Detection

Classifies a PDF as one of three types based on its extractable text layer:
- "text"    — >80% of pages have meaningful text (>50 chars)
- "scanned" — <20% of pages have text (image-only PDF)
- "hybrid"  — mixed content (everything else)

Writes a classification record to data/type_map.jsonl.
"""

import argparse
import hashlib
import json
from pathlib import Path

import fitz  # PyMuPDF


REGISTRY_PATH = Path("data") / "registry.jsonl"
TYPE_MAP_PATH = Path("data") / "type_map.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_sha256(filepath: Path) -> str:
    """SHA-256 hex digest, streamed for large files."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_registry() -> list[dict]:
    """Read all records from the JSONL registry (empty if missing)."""
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
    """Look up the document_id for a given SHA-256 hash."""
    for r in registry:
        if r.get("sha256") == sha256:
            return r["document_id"]
    return None


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------


def detect_type(pdf_path: str) -> dict:
    """Analyse a PDF and return its type classification record."""
    path = Path(pdf_path).resolve()

    if not path.exists():
        print(f"❌  Error: File not found — {path}")
        raise SystemExit(1)

    if path.suffix.lower() != ".pdf":
        print(f"❌  Error: Not a PDF file — {path.suffix}")
        raise SystemExit(1)

    # Resolve document_id from the registry
    sha256 = _compute_sha256(path)
    registry = _load_registry()
    doc_id = _find_document_id(sha256, registry)

    if doc_id is None:
        print(
            "❌  Error: PDF not found in registry. "
            "Run phase_1_1_register.py first."
        )
        raise SystemExit(1)

    # --- Per-page text analysis --------------------------------------------
    doc = fitz.open(str(path))
    try:
        total_pages = doc.page_count
        if total_pages == 0:
            print("❌  Error: PDF has zero pages.")
            raise SystemExit(1)

        text_page_count = 0
        for page in doc:
            text = page.get_text().strip()
            if len(text) > 50:
                text_page_count += 1
    finally:
        doc.close()

    text_ratio = text_page_count / total_pages

    # --- Classification ----------------------------------------------------
    if text_ratio >= 0.8:
        pdf_type = "text"
        confidence = text_ratio
        strategy = "pymupdf"
    elif text_ratio <= 0.2:
        pdf_type = "scanned"
        confidence = 1.0 - text_ratio
        strategy = "ocr"
    else:
        pdf_type = "hybrid"
        # Confidence peaks at 1.0 when the split is exactly 50/50
        confidence = 1.0 - 2.0 * abs(text_ratio - 0.5)
        strategy = "mixed"

    return {
        "document_id": doc_id,
        "sha256": sha256,
        "filename": path.name,
        "page_count": total_pages,
        "pdf_type": pdf_type,
        "confidence": round(confidence, 4),
        "strategy": strategy,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.2 — Detect PDF type (text / scanned / hybrid)."
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    result = detect_type(args.pdf_path)

    TYPE_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TYPE_MAP_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(
        f"✅  Type: {result['pdf_type']}  |  "
        f"{result['filename']}  |  "
        f"{result['page_count']} pages  |  "
        f"confidence: {result['confidence']:.2f}  |  "
        f"strategy: {result['strategy']}"
    )


if __name__ == "__main__":
    main()
