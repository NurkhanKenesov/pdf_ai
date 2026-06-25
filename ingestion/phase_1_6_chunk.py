#!/usr/bin/env python3
"""
Phase 1.6 — Semantic Chunking

Splits normalised chunks into semantically-coherent chunks via a sliding
window with paragraph-boundary respect and configurable overlap.

- Token counting: whitespace splitting (no tiktoken)
- Target size: 400 tokens, overlap: 50 tokens
- Paragraph boundaries (double newline) are never split
- Single paragraphs larger than the target are split at sentence boundaries
  (". " or ".\\n")

Input:  data/normalized_chunks.jsonl  (filtered by --doc_id)
Output: data/semantic_chunks.jsonl
"""

import argparse
import json
import re
import uuid
from pathlib import Path


NORMALIZED_PATH = Path("data") / "normalized_chunks.jsonl"
SEMANTIC_PATH = Path("data") / "semantic_chunks.jsonl"

TARGET_TOKENS = 400
OVERLAP_TOKENS = 50

# Sentence boundary: ". " or ".\n" — lookbehind keeps the period on each fragment
SENTENCE_BOUNDARY = re.compile(r"(?<=\.)\s+")


# ---------------------------------------------------------------------------
# Token counting (stdlib only)
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    """Approximate token count via whitespace splitting (1 token ≈ 1 word)."""
    return len(text.split()) if text.strip() else 0


# ---------------------------------------------------------------------------
# Leaf-level helpers
# ---------------------------------------------------------------------------


def _split_paragraph_at_sentences(text: str) -> list[str]:
    """
    Split a single paragraph at sentence boundaries (". " or ".\\n").

    Uses a lookbehind so the period stays attached to each sentence.
    Returns a list of sentence strings (trailing whitespace stripped).
    """
    parts = SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in parts if s.strip()]


def _make_chunk_record(
    text: str,
    token_count: int,
    doc_id: str,
    pages: set[int],
) -> dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "doc_id": doc_id,
        "source_pages": sorted(pages),
        "token_count": token_count,
        "text": text,
    }


# ---------------------------------------------------------------------------
# Paragraph extraction from normalised chunks
# ---------------------------------------------------------------------------


def _load_paragraphs(doc_id: str) -> list[tuple[str, int, int]]:
    """
    Load all normalised chunks for *doc_id* and explode into paragraphs.

    Returns list of (paragraph_text, token_count, page_number).
    """
    if not NORMALIZED_PATH.exists():
        print(f"❌  Error: {NORMALIZED_PATH} not found. Run phase_1_5 first.")
        raise SystemExit(1)

    paragraphs: list[tuple[str, int, int]] = []
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

            page = record.get("page", 0)
            text = record.get("text", "").strip()
            if not text:
                continue

            # Split this page's text into paragraphs (double newline)
            page_paragraphs = re.split(r"\n\s*\n", text)
            # Fallback: if no double-newline found, split on single newline
            if len(page_paragraphs) <= 1 and "\n" in text:
                page_paragraphs = re.split(r"\n", text)
            for para in page_paragraphs:
                para = para.strip()
                if not para:
                    continue
                tok = _count_tokens(para)
                paragraphs.append((para, tok, page))

    if not paragraphs:
        print(f"❌  Error: No normalised chunks found for doc_id {doc_id}")
        raise SystemExit(1)

    return paragraphs


# ---------------------------------------------------------------------------
# Sliding-window chunking
# ---------------------------------------------------------------------------


def _chunk_large_paragraph(
    text: str,
    tokens: int,
    page: int,
    doc_id: str,
    chunks_out: list[dict],
) -> int:
    """
    Split a single paragraph larger than *TARGET_TOKENS* at sentence
    boundaries, producing multiple chunks.

    Returns the last overlap token count so the caller can continue cleanly
    (unused in current call chain but kept for consistency).
    """
    sentences = _split_paragraph_at_sentences(text)
    if not sentences:
        return 0

    # If individual sentences are still huge, fall back to word-level grouping
    buffer: list[str] = []
    buffer_tok = 0
    for sent in sentences:
        sent_tok = _count_tokens(sent)
        if sent_tok > TARGET_TOKENS:
            # Sentence itself is larger than target — split by words
            words = sent.split()
            word_group: list[str] = []
            group_tok = 0
            for w in words:
                if group_tok + 1 > TARGET_TOKENS and word_group:
                    chunks_out.append(
                        _make_chunk_record(
                            " ".join(word_group), group_tok, doc_id, {page}
                        )
                    )
                    # Overlap: carry last ~OVERLAP_TOKENS words
                    carry_tok = 0
                    carry_words: list[str] = []
                    for cw in reversed(word_group):
                        if carry_tok + 1 > OVERLAP_TOKENS:
                            break
                        carry_words.insert(0, cw)
                        carry_tok += 1
                    word_group = list(carry_words)
                    group_tok = carry_tok
                word_group.append(w)
                group_tok += 1
            if word_group:
                chunks_out.append(
                    _make_chunk_record(
                        " ".join(word_group), group_tok, doc_id, {page}
                    )
                )
            continue

        if buffer_tok + sent_tok > TARGET_TOKENS and buffer:
            chunks_out.append(
                _make_chunk_record(
                    " ".join(buffer), buffer_tok, doc_id, {page}
                )
            )
            # Overlap: carry last ~OVERLAP_TOKENS of buffer
            carry_tok = 0
            carry: list[str] = []
            for bs in reversed(buffer):
                bst = _count_tokens(bs)
                if carry_tok + bst > OVERLAP_TOKENS:
                    break
                carry.insert(0, bs)
                carry_tok += bst
            buffer = list(carry)
            buffer_tok = carry_tok

        buffer.append(sent)
        buffer_tok += sent_tok

    if buffer:
        chunks_out.append(
            _make_chunk_record(" ".join(buffer), buffer_tok, doc_id, {page})
        )

    return buffer_tok


def build_semantic_chunks(
    paragraphs: list[tuple[str, int, int]], doc_id: str
) -> list[dict]:
    """
    Build semantic chunks from a list of (text, token_count, page) tuples.

    Algorithm:
    - Accumulate paragraphs respecting paragraph boundaries.
    - Once the token budget (target) is reached, emit a chunk.
    - Overlap: the next chunk starts from the paragraph that pushed
      cumulative tokens past (target - overlap).
    - Paragraphs larger than the target are split at sentence boundaries.
    """
    chunks: list[dict] = []
    idx = 0

    while idx < len(paragraphs):
        chunk_parts: list[str] = []
        chunk_tokens = 0
        chunk_pages: set[int] = set()
        overlap_next = idx  # default: start next chunk here (no overlap)

        j = idx
        while j < len(paragraphs):
            text, tokens, page = paragraphs[j]

            # Single paragraph exceeds target — delegate to sentence splitting
            if tokens > TARGET_TOKENS and not chunk_parts:
                _chunk_large_paragraph(text, tokens, page, doc_id, chunks)
                j += 1
                idx = j
                break

            # Does this paragraph fit in the current budget?
            if chunk_tokens + tokens <= TARGET_TOKENS:
                chunk_parts.append(text)
                chunk_tokens += tokens
                chunk_pages.add(page)

                # Record where the overlap region starts
                if chunk_tokens >= TARGET_TOKENS - OVERLAP_TOKENS:
                    if overlap_next == idx:
                        overlap_next = j + 1  # next paragraph begins overlap

                j += 1
            else:
                # Paragraph doesn't fit — finalise current chunk
                break

        # ── If we accumulated any text, emit a chunk ──────────────────────
        if chunk_parts:
            chunks.append(
                _make_chunk_record(
                    "\n\n".join(chunk_parts), chunk_tokens, doc_id, chunk_pages
                )
            )

        # ── Advance index ─────────────────────────────────────────────────
        if overlap_next > idx:
            idx = overlap_next
        else:
            idx = j

        # Prevent infinite loop if no progress was made
        if idx <= (chunks[-1]["token_count"] if chunks else -1) and not chunk_parts:
            idx = j + 1

        if idx >= len(paragraphs):
            break

    # ── Post-processing: merge chunks under 30 tokens into the next ─────
    merged: list[dict] = []
    for i, chunk in enumerate(chunks):
        if chunk["token_count"] < 30 and i < len(chunks) - 1:
            # Merge into the next chunk unconditionally
            nxt = chunks[i + 1]
            nxt["text"] = chunk["text"] + "\n\n" + nxt["text"]
            nxt["token_count"] = chunk["token_count"] + nxt["token_count"]
            nxt["source_pages"] = sorted(
                set(chunk["source_pages"] + nxt["source_pages"])
            )
        else:
            merged.append(chunk)

    return merged


# ---------------------------------------------------------------------------
# Passthrough mode (short documents)
# ---------------------------------------------------------------------------


def _passthrough_chunks(
    paragraphs: list[tuple[str, int, int]], doc_id: str
) -> list[dict]:
    """
    Group input paragraphs by page and emit one semantic chunk per page.

    Used when the sliding window would produce a single under-split chunk
    for a document with 3+ pages — ensures each page becomes its own chunk.
    """
    from collections import OrderedDict

    page_groups: dict[int, list[str]] = OrderedDict()
    for text, _tokens, page in paragraphs:
        page_groups.setdefault(page, []).append(text)

    chunks: list[dict] = []
    for page, texts in page_groups.items():
        page_text = "\n\n".join(texts)
        tok = _count_tokens(page_text)
        chunks.append(
            _make_chunk_record(page_text, tok, doc_id, {page})
        )
    return chunks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def chunk_document(doc_id: str) -> int:
    """Run the full chunking pipeline for *doc_id*. Returns chunk count."""
    paragraphs = _load_paragraphs(doc_id)
    chunks = build_semantic_chunks(paragraphs, doc_id)

    # ── Passthrough mode for short documents ────────────────────────────
    # If the sliding window collapsed everything into 1 chunk but the input
    # had 3+ source pages, skip the window and output each page as its own
    # semantic chunk. This prevents tiny documents from being under-split.
    unique_pages = len(set(p[2] for p in paragraphs))
    if len(chunks) == 1 and unique_pages >= 3:
        chunks = _passthrough_chunks(paragraphs, doc_id)

    if not chunks:
        print(f"⚠️  No semantic chunks generated for {doc_id}")
        return 0

    SEMANTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEMANTIC_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    total_tokens = sum(c["token_count"] for c in chunks)
    print(
        f"✅  Chunked: {doc_id}  |  "
        f"{len(chunks)} semantic chunks  |  "
        f"~{total_tokens} total tokens  |  "
        f"avg {total_tokens // max(len(chunks), 1)} tok/chunk"
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.6 — Build semantic chunks with sliding-window overlap."
    )
    parser.add_argument("--doc_id", type=str, required=True, help="Document UUID")
    args = parser.parse_args()
    chunk_document(args.doc_id)


if __name__ == "__main__":
    main()
