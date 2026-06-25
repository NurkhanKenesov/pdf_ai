#!/usr/bin/env python3
"""
Phase 1.1 — PDF Reception and Registration

CLI script that registers a PDF file into the ENT ingestion pipeline:
- Extracts metadata (page count, language, file size, SHA-256 hash, filename)
- Assigns a unique document_id (UUID4)
- Appends the registration record to data/registry.jsonl
"""

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

import fitz  # PyMuPDF


# Kazakh-specific Cyrillic characters (both lower and upper case)
KAZAKH_LETTERS = set("әіңғүұқөһӘІҢҒҮҰҚӨҺ")

# Registry file path is relative to the current working directory
REGISTRY_PATH = Path("data") / "registry.jsonl"


# ---------------------------------------------------------------------------
# Metadata extraction helpers
# ---------------------------------------------------------------------------


def compute_sha256(filepath: Path) -> str:
    """SHA-256 hex digest of a file, streamed to handle large PDFs."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_page_count(filepath: Path) -> int:
    """Return the number of pages in a PDF using PyMuPDF."""
    doc = fitz.open(str(filepath))
    try:
        return doc.page_count
    finally:
        doc.close()


def detect_language(filepath: Path) -> str:
    """
    Heuristic language detection based on Kazakh-specific letter frequency.

    Extracts text from every page.  If Kazakh-specific letters (ә, і, ң, ғ,
    ү, ұ, қ, ө, һ and their uppercase variants) make up > 3 % of all
    characters in the document, classify as "kz"; otherwise "ru".

    Falls back to "ru" if the PDF yields no extractable text.
    """
    doc = fitz.open(str(filepath))
    try:
        text_parts = [page.get_text() for page in doc]
    finally:
        doc.close()

    text = "".join(text_parts)
    if not text.strip():
        return "ru"

    total = len(text)
    kazakh_count = sum(1 for ch in text if ch in KAZAKH_LETTERS)
    return "kz" if (kazakh_count / total) > 0.03 else "ru"


def load_registry() -> list[dict]:
    """Read every record currently in the JSONL registry (empty list if missing)."""
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
                # Silently skip corrupted lines so a bad write doesn't
                # break future registrations.
                continue
    return records


def is_already_registered(sha256: str, registry: list[dict]) -> bool:
    """Return True if *sha256* already exists in the registry."""
    return any(r.get("sha256") == sha256 for r in registry)


# ---------------------------------------------------------------------------
# Main registration logic
# ---------------------------------------------------------------------------


def register_pdf(pdf_path: str) -> None:
    """Validate a PDF, extract its metadata, and persist a registration record."""
    path = Path(pdf_path).resolve()

    # --- Input validation ---------------------------------------------------
    if not path.exists():
        print(f"❌  Error: File not found — {path}")
        raise SystemExit(1)

    if path.suffix.lower() != ".pdf":
        print(f"❌  Error: Not a PDF file — {path.suffix}")
        raise SystemExit(1)

    # --- Compute hash & check for duplicates --------------------------------
    sha256 = compute_sha256(path)
    registry = load_registry()

    if is_already_registered(sha256, registry):
        print(
            f"⚠️  Skipping — already registered (SHA-256: {sha256[:16]}…)  "
            f"{path.name}"
        )
        raise SystemExit(0)

    # --- Extract PDF metadata -----------------------------------------------
    try:
        page_count = get_page_count(path)
    except Exception as exc:
        print(f"❌  Error: Corrupted or unreadable PDF — {exc}")
        raise SystemExit(1) from exc

    try:
        lang = detect_language(path)
    except Exception as exc:
        print(f"❌  Error: Failed to extract text from PDF — {exc}")
        raise SystemExit(1) from exc

    file_size_kb = path.stat().st_size / 1024.0

    # --- Build & persist registration record --------------------------------
    document_id = str(uuid.uuid4())
    record = {
        "document_id": document_id,
        "filename": path.name,
        "sha256": sha256,
        "page_count": page_count,
        "language": lang,
        "file_size_kb": round(file_size_kb, 2),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- User-facing confirmation -------------------------------------------
    print(
        f"✅  Registered: {document_id}  |  "
        f"{path.name}  |  "
        f"{page_count} pages  |  "
        f"lang: {lang}"
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.1 — Register a PDF file in the ENT ingestion pipeline."
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF file to register",
    )
    args = parser.parse_args()
    register_pdf(args.pdf_path)


if __name__ == "__main__":
    main()
