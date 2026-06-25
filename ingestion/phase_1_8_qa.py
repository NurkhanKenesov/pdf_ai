#!/usr/bin/env python3
"""
Phase 1.8 — Quality Assurance

Applies a set of rejection rules to enriched chunks and separates them into
passed / rejected streams.

Rejection rules (ANY match → reject):
1. token_count < 50  (chunks with fewer than 50 tokens are too short to be useful)
2. text is empty or whitespace-only
3. >40 % non-Cyrillic / non-Latin / non-digit characters (OCR garbage)
4. doc_id or source_pages missing or null
5. text contains only repeated characters or symbols  (r'(.)\\1{10,}')

Input:  data/enriched_chunks.jsonl  (filtered by --doc_id)
Output: data/qa_passed.jsonl  +  data/qa_rejected.jsonl
"""

import argparse
import json
import re
from pathlib import Path


ENRICHED_PATH = Path("data") / "enriched_chunks.jsonl"
QA_PASSED_PATH = Path("data") / "qa_passed.jsonl"
QA_REJECTED_PATH = Path("data") / "qa_rejected.jsonl"

# Longer-than-expected repeats of the same character — likely garbage
REPEATED_CHAR = re.compile(r"(.)\1{10,}")

# Characters considered "clean" for the OCR-garbage heuristic.
# Everything outside these sets counts toward the garbage ratio.
# (Using double-quote-safe format by encoding the inner quote char via chr)
_CLEAN_PATTERN = re.compile(
    "["
    "\u0400-\u04FF"          # Cyrillic
    "\u0500-\u052F"          # Cyrillic Supplement
    "A-Za-z"
    "0-9"
    "\\s"
    ".,;:!?\\-()\\[\\]{}"  # note: double-quote opens a new string; chr(34) is "
    + chr(34) + "'"  # literal " and '
    + """`«»/@#$%^&*~<>=+\u007C—–…№]"""
).match


def _is_ocr_garbage(text: str) -> bool:
    """
    Return True if >40 % of characters are outside the clean set
    (Cyrillic, Latin, digits, whitespace, common punctuation).
    """
    if not text.strip():
        return True
    garbage = sum(1 for c in text if not _CLEAN_PATTERN(c))
    return (garbage / len(text)) > 0.40


def _check_repeated_chars(text: str) -> bool:
    """
    Return True if the text consists mostly of one repeated character,
    e.g. "--------" or "aaaaaaaaaaaa".
    """
    stripped = text.strip()
    if not stripped:
        return False
    return bool(REPEATED_CHAR.search(stripped))


# ---------------------------------------------------------------------------
# QA orchestration
# ---------------------------------------------------------------------------


def qa_filter(doc_id: str) -> tuple[int, int]:
    """
    Run QA rules on all enriched chunks for *doc_id*.

    Returns (passed_count, rejected_count).
    """
    if not ENRICHED_PATH.exists():
        print(f"❌  Error: {ENRICHED_PATH} not found. Run phase_1_7 first.")
        raise SystemExit(1)

    passed: list[dict] = []
    rejected: list[dict] = []

    with open(ENRICHED_PATH, "r", encoding="utf-8") as f:
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

            # ── Rule checks ───────────────────────────────────────────────
            reject_reason: str | None = None

            # Rule 4: Missing doc_id or source_pages
            if not chunk.get("doc_id") or chunk.get("source_pages") is None:
                reject_reason = "missing doc_id or source_pages"

            # Rule 2: Empty / whitespace-only text
            if reject_reason is None:
                text = chunk.get("text", "")
                if not text or not text.strip():
                    reject_reason = "empty or whitespace-only text"

            # Rule 1: token_count < 50  (too short to be useful as context)
            if reject_reason is None:
                token_count = chunk.get("token_count", 0)
                if token_count < 50:
                    reject_reason = "token_count < 50"

            # Rule 5: Repeated characters
            if reject_reason is None:
                text = chunk.get("text", "")
                if _check_repeated_chars(text):
                    reject_reason = "repeated characters"

            # Rule 3: OCR garbage (>40 % non-clean chars)
            if reject_reason is None:
                text = chunk.get("text", "")
                if _is_ocr_garbage(text):
                    reject_reason = "OCR garbage (>40% non-clean chars)"

            # ── Route ─────────────────────────────────────────────────────
            if reject_reason:
                rejected.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                    "reject_reason": reject_reason,
                    "token_count": chunk.get("token_count"),
                    "source_pages": chunk.get("source_pages"),
                })
            else:
                passed.append(chunk)

    # Write passed chunks
    QA_PASSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QA_PASSED_PATH, "a", encoding="utf-8") as f:
        for rec in passed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write rejected chunks
    with open(QA_REJECTED_PATH, "a", encoding="utf-8") as f:
        for rec in rejected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(passed) + len(rejected)
    print(
        f"✅  QA complete: {len(passed)} passed, "
        f"{len(rejected)} rejected out of {total}"
    )
    return len(passed), len(rejected)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.8 — QA filter for enriched chunks."
    )
    parser.add_argument("--doc_id", type=str, required=True, help="Document UUID")
    args = parser.parse_args()
    qa_filter(args.doc_id)


if __name__ == "__main__":
    main()
