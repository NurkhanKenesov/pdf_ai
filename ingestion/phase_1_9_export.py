#!/usr/bin/env python3
"""
Phase 1.9 — Export

Validates QA-passed chunks against the final schema and writes clean
records to ``data/final_chunks.jsonl`` — the handoff to Phase 2 (Indexing).

Schema (all fields required, use ``None`` for missing):
- chunk_id: str (UUID4)
- doc_id: str (UUID4)
- text: str
- token_count: int
- source_pages: list[int]
- publisher: str
- book_title: str
- language: str  ("ru" | "kz")
- chapter: str | None
- formula_present: bool
- table_present: bool
- has_image: bool
- image_path: str | None

Input:  data/qa_passed.jsonl  (filtered by --doc_id)
Output: data/final_chunks.jsonl  +  data/export_errors.jsonl
"""

import argparse
import json
from pathlib import Path


QA_PASSED_PATH = Path("data") / "qa_passed.jsonl"
FINAL_PATH = Path("data") / "final_chunks.jsonl"
ERRORS_PATH = Path("data") / "export_errors.jsonl"

# The exact set of fields expected in the final output
REQUIRED_FIELDS = {
    "chunk_id": str,
    "doc_id": str,
    "text": str,
    "token_count": int,
    "source_pages": list,
    "publisher": str,
    "book_title": str,
    "language": str,
    "chapter": (str, type(None)),
    "formula_present": bool,
    "table_present": bool,
    "has_image": bool,
    "image_path": (str, type(None)),
}

VALID_LANGUAGES = {"ru", "kz"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(record: dict, index: int) -> str | None:
    """
    Validate *record* against the final schema.

    Returns an error message string on failure, or None on success.
    """
    # ── Check for missing fields ──────────────────────────────────────────
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in record:
            return f"line {index}: missing field '{field}'"
        value = record[field]

        # Type check: expected_type may be a tuple of acceptable types
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                return (
                    f"line {index}: field '{field}' has wrong type "
                    f"(expected one of {expected_type}, got {type(value).__name__})"
                )
        else:
            if not isinstance(value, expected_type):
                return (
                    f"line {index}: field '{field}' has wrong type "
                    f"(expected {expected_type.__name__}, got {type(value).__name__})"
                )

    # ── Specific value constraints ────────────────────────────────────────
    if not isinstance(record["source_pages"], list):
        return f"line {index}: 'source_pages' must be a list"

    if record["language"] not in VALID_LANGUAGES:
        return (
            f"line {index}: invalid language '{record['language']}' "
            f"(expected one of {VALID_LANGUAGES})"
        )

    if not isinstance(record["token_count"], int) or record["token_count"] < 0:
        return f"line {index}: 'token_count' must be a non-negative integer"

    if not record.get("chunk_id"):
        return f"line {index}: 'chunk_id' must be non-empty"

    if not record.get("doc_id"):
        return f"line {index}: 'doc_id' must be non-empty"

    return None


# ---------------------------------------------------------------------------
# Export orchestration
# ---------------------------------------------------------------------------


def export_final(doc_id: str) -> int:
    """
    Validate and export all QA-passed chunks for *doc_id*.

    Returns the number of successfully written records.
    """
    if not QA_PASSED_PATH.exists():
        print(f"❌  Error: {QA_PASSED_PATH} not found. Run phase_1_8 first.")
        raise SystemExit(1)

    written = 0
    errors: list[dict] = []

    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(QA_PASSED_PATH, "r", encoding="utf-8") as fin,
        open(FINAL_PATH, "a", encoding="utf-8") as fout,
    ):
        for line_no, line in enumerate(fin, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if record.get("doc_id") != doc_id:
                    continue
            except json.JSONDecodeError:
                errors.append({
                    "line": line_no,
                    "error": "invalid JSON",
                    "raw": stripped[:200],
                })
                continue

            # ── Ensure all fields are present (fill missing with nulls) ───
            cleaned = {field: record.get(field) for field in REQUIRED_FIELDS}

            # ── Validate ──────────────────────────────────────────────────
            error_msg = _validate(cleaned, line_no)
            if error_msg:
                errors.append({
                    "line": line_no,
                    "error": error_msg,
                    "chunk_id": cleaned.get("chunk_id"),
                })
                continue

            # ── Write ─────────────────────────────────────────────────────
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            written += 1

    # Write errors report
    if errors:
        with open(ERRORS_PATH, "a", encoding="utf-8") as ferr:
            for err in errors:
                ferr.write(json.dumps(err, ensure_ascii=False) + "\n")

    print(f"✅  Export complete: {written} chunks written to data/final_chunks.jsonl")
    if errors:
        print(f"⚠️  {len(errors)} records logged to data/export_errors.jsonl")

    return written


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.9 — Validate and export final chunks for indexing."
    )
    parser.add_argument("--doc_id", type=str, required=True, help="Document UUID")
    args = parser.parse_args()
    export_final(args.doc_id)


if __name__ == "__main__":
    main()
