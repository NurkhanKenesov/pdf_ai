#!/usr/bin/env python3
"""
Phase 1.7 — Enrichment

Adds publisher, book title, language, chapter/section heading, and
aggregated page-level metadata (formula_present, table_present, has_image,
image_path) to each semantic chunk.

Input:  data/semantic_chunks.jsonl + data/registry.jsonl + data/normalized_chunks.jsonl
        (all filtered by --doc_id)
Output: data/enriched_chunks.jsonl
"""

import argparse
import json
import re
from pathlib import Path


REGISTRY_PATH = Path("data") / "registry.jsonl"
NORMALIZED_PATH = Path("data") / "normalized_chunks.jsonl"
SEMANTIC_PATH = Path("data") / "semantic_chunks.jsonl"
ENRICHED_PATH = Path("data") / "enriched_chunks.jsonl"

# Patterns for section headings (Kazakh, Russian, Latin §)
CHAPTER_PATTERNS = [
    re.compile(r"^§\s*\d+[\s.].*$", re.MULTILINE),
    re.compile(r"^Глава\s+\d+[\s.].*$", re.MULTILINE),
    re.compile(r"^Тарау\s+\d+[\s.].*$", re.MULTILINE),
    re.compile(r"^Б[өо]лім\s+\d+[\s.].*$", re.MULTILINE),
]

# Map filename keywords to publisher name
PUBLISHER_MAP = {
    "mektep": "Мектеп",
    "atamura": "Атамұра",
    "arman": "Арман-ПВ",
}


# ---------------------------------------------------------------------------
# Publisher derivation
# ---------------------------------------------------------------------------


def _derive_publisher(filename: str) -> str:
    """Derive publisher from the filename using known keywords."""
    lower = filename.lower()
    for keyword, publisher in PUBLISHER_MAP.items():
        if keyword in lower:
            return publisher
    return "unknown"


# ---------------------------------------------------------------------------
# Chapter detection
# ---------------------------------------------------------------------------


def _detect_chapter(text: str) -> str | None:
    """
    Scan *text* for section headings (§ N, Глава N, Тарау N, Бөлім N).

    Returns the first matched heading, or None.
    """
    for pattern in CHAPTER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    return None


# ---------------------------------------------------------------------------
# Per-page metadata index (from normalised chunks)
# ---------------------------------------------------------------------------


def _build_page_index(doc_id: str) -> dict[int, dict]:
    """
    Build a {page_num: {formula_present, table_present, has_image, image_path}}
    index from ``data/normalized_chunks.jsonl`` for *doc_id*.
    """
    index: dict[int, dict] = {}
    if not NORMALIZED_PATH.exists():
        return index

    with open(NORMALIZED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if record.get("doc_id") != doc_id:
                    continue
            except json.JSONDecodeError:
                continue

            page = record.get("page")
            if page is None:
                continue
            index[page] = {
                "formula_present": record.get("formula_present", False),
                "table_present": record.get("table_present", False),
                "has_image": record.get("has_image", False),
                "image_path": record.get("image_path"),
            }
    return index


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _find_registry_record(doc_id: str) -> dict | None:
    """Find the registry record for *doc_id*."""
    if not REGISTRY_PATH.exists():
        return None
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if record.get("document_id") == doc_id:
                    return record
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Enrichment orchestration
# ---------------------------------------------------------------------------


def enrich(doc_id: str, publisher_override: str | None = None) -> int:
    """Enrich all semantic chunks for *doc_id*. Returns enriched count."""
    if not SEMANTIC_PATH.exists():
        print(f"❌  Error: {SEMANTIC_PATH} not found. Run phase_1_6 first.")
        raise SystemExit(1)

    # ── Registry metadata ─────────────────────────────────────────────────
    registry_record = _find_registry_record(doc_id)
    if registry_record is None:
        print(f"❌  Error: doc_id {doc_id} not found in registry.")
        raise SystemExit(1)

    filename = registry_record.get("filename", "")
    language = registry_record.get("language", "ru")
    publisher = _derive_publisher(filename)

    # Fallback: CLI override
    if publisher == "unknown" and publisher_override is not None:
        publisher = publisher_override

    if publisher == "unknown":
        print(
            f"⚠️  Publisher unknown. Re-run with --publisher \"Мектеп\""
        )

    book_title = Path(filename).stem  # filename without extension

    # ── Per-page metadata index ───────────────────────────────────────────
    page_index = _build_page_index(doc_id)

    # ── Load & enrich semantic chunks ─────────────────────────────────────
    enriched: list[dict] = []
    with open(SEMANTIC_PATH, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                chunk = json.loads(stripped)
                if chunk.get("doc_id") != doc_id:
                    continue
            except json.JSONDecodeError:
                continue

            source_pages: list[int] = chunk.get("source_pages", [])

            # Aggregate per-page booleans across all source pages
            formula = any(
                page_index.get(p, {}).get("formula_present", False)
                for p in source_pages
            )
            table = any(
                page_index.get(p, {}).get("table_present", False)
                for p in source_pages
            )
            image = any(
                page_index.get(p, {}).get("has_image", False)
                for p in source_pages
            )
            # Use the first non-null image_path found
            image_path: str | None = None
            for p in source_pages:
                ip = page_index.get(p, {}).get("image_path")
                if ip is not None:
                    image_path = ip
                    break

            # Chapter detection
            chapter = _detect_chapter(chunk.get("text", ""))

            enriched.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": doc_id,
                "text": chunk["text"],
                "token_count": chunk["token_count"],
                "source_pages": chunk["source_pages"],
                "publisher": publisher,
                "book_title": book_title,
                "language": language,
                "chapter": chapter,
                "formula_present": formula,
                "table_present": table,
                "has_image": image,
                "image_path": image_path,
            })

    if not enriched:
        print(f"⚠️  No semantic chunks found for doc_id {doc_id}")
        return 0

    ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENRICHED_PATH, "a", encoding="utf-8") as f:
        for rec in enriched:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    chapters = sum(1 for r in enriched if r["chapter"] is not None)
    print(
        f"✅  Enriched: {doc_id}  |  "
        f"{len(enriched)} chunks  |  "
        f"publisher: {publisher}  |  "
        f"lang: {language}  |  "
        f"{chapters} with chapter heading"
    )
    return len(enriched)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.7 — Enrich chunks with publisher, chapter, page metadata."
    )
    parser.add_argument("--doc_id", type=str, required=True, help="Document UUID")
    parser.add_argument("--publisher", type=str, default=None,
                        help="Override publisher name (e.g. \"Мектеп\")")
    args = parser.parse_args()
    enrich(args.doc_id, publisher_override=args.publisher)


if __name__ == "__main__":
    main()
