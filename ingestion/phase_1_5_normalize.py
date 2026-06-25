#!/usr/bin/env python3
"""
Phase 1.5 — Normalization

Normalises extracted text chunks by:
- Replacing math/formula sequences with [FORMULA: <raw>] tokens
- Converting tab/space-aligned table text to Markdown table format
- Sorting chunks by page number ascending

Input:  data/chunks.jsonl  (filtered by --doc_id)
Output: data/normalized_chunks.jsonl  (same schema + "normalized": true)
"""

import argparse
import json
import re
from pathlib import Path


CHUNKS_PATH = Path("data") / "chunks.jsonl"
NORMALIZED_PATH = Path("data") / "normalized_chunks.jsonl"

# Pattern matching math-like sequences:
#   - digit±operator±digit       (e.g. 3+5, 2x−1)
#   - Greek letters (Unicode)
#   - standalone math symbols    (√ ∫ ² ³ = etc.)
#   - 2+ consecutive math chars  from the operator set
MATH_SEQUENCE = re.compile(
    r"\d+\s*[+\-−×÷*/^]\s*\d+"                     # digit-operator-digit
    r"|[\u0391-\u03FF\u03B1-\u03C9]"               # Greek uppercase + lowercase
    r"|[√∫²³]"
    r"|[=\+\-*/^√∫²³αβγδεζθλμπσφω]{2,}"           # 2+ consecutive math chars
)

# Two or more consecutive spaces — used as column delimiter
COLUMN_DELIM = re.compile(r"  +")


# ---------------------------------------------------------------------------
# Formula tokenisation
# ---------------------------------------------------------------------------


def _replace_formulas(text: str) -> str:
    """Replace every matched math sequence with ``[FORMULA: <raw>]``."""
    def _replacer(m: re.Match) -> str:
        raw = m.group(0).strip()
        return f"[FORMULA: {raw}]"

    return MATH_SEQUENCE.sub(_replacer, text)


# ---------------------------------------------------------------------------
# Table → Markdown conversion
# ---------------------------------------------------------------------------


def _to_markdown_table(text: str) -> str:
    """
    Convert tab/space-aligned text to a Markdown table.

    - Rows are separated by newlines.
    - Columns are separated by 2+ consecutive spaces or tabs.
    - First row becomes the header; a separator row is inserted.
    """
    lines = text.strip().split("\n")
    if not lines:
        return text

    rows: list[list[str]] = []
    for line in lines:
        cells = [c.strip() for c in COLUMN_DELIM.split(line) if c.strip()]
        if not cells:
            continue
        # Also split on tabs
        if "\t" in line:
            cells = [c.strip() for c in line.split("\t") if c.strip()]
        if cells:
            rows.append(cells)

    if len(rows) < 2:
        # Not enough rows for a meaningful table — leave as-is
        return text

    # Normalise column count
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    # Build Markdown
    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join("---" for _ in range(max_cols)) + " |"
    body_lines = ["| " + " | ".join(r) + " |" for r in rows[1:]]

    return "\n".join([header, separator] + body_lines)


# ---------------------------------------------------------------------------
# Normalisation orchestration
# ---------------------------------------------------------------------------


def normalize(doc_id: str) -> int:
    """Normalise all chunks belonging to *doc_id*. Returns chunk count."""
    if not CHUNKS_PATH.exists():
        print(f"❌  Error: {CHUNKS_PATH} not found. Run phase_1_4 first.")
        raise SystemExit(1)

    chunks: list[dict] = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if record.get("doc_id") == doc_id:
                    chunks.append(record)
            except json.JSONDecodeError:
                continue

    if not chunks:
        print(f"❌  Error: No chunks found for doc_id {doc_id}")
        raise SystemExit(1)

    # Sort by page ascending
    chunks.sort(key=lambda c: c["page"])

    normalized: list[dict] = []
    for chunk in chunks:
        text = chunk["text"]

        if chunk.get("formula_present"):
            text = _replace_formulas(text)

        if chunk.get("table_present"):
            text = _to_markdown_table(text)

        record = dict(chunk)
        record["text"] = text
        record["normalized"] = True
        normalized.append(record)

    NORMALIZED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NORMALIZED_PATH, "w", encoding="utf-8") as f:
        for rec in normalized:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    formulas = sum(1 for r in normalized if r["formula_present"])
    tables = sum(1 for r in normalized if r["table_present"])
    print(
        f"✅  Normalized: {doc_id}  |  "
        f"{len(normalized)} chunks  |  "
        f"{formulas} formula pages  |  "
        f"{tables} table pages"
    )
    return len(normalized)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.5 — Normalize chunks (formulas, tables, sort by page)."
    )
    parser.add_argument("--doc_id", type=str, required=True, help="Document UUID")
    args = parser.parse_args()
    normalize(args.doc_id)


if __name__ == "__main__":
    main()
