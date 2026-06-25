#!/usr/bin/env python3
"""
Phase 2.3 — BM25 Indexing

Builds one BM25Okapi index per publisher from the final chunks.

- Tokenizer: whitespace split + lowercase + strip punctuation
- Serialises each index via pickle: data/bm25_{publisher_slug}.pkl
- Saves corpus metadata as JSON:   data/bm25_{publisher_slug}_meta.json
- Idempotent: loads existing index, merges new chunks, resaves
"""

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi


FINAL_PATH = Path("data") / "final_chunks.jsonl"
BM25_DIR = Path("data")

# Regex to strip punctuation for tokenization
PUNCT_RE = re.compile(r"[^\w\s]")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Whitespace-split, lowercase, strip punctuation."""
    return PUNCT_RE.sub("", text.lower()).split()


# ---------------------------------------------------------------------------
# Load existing BM25 index + meta (if present)
# ---------------------------------------------------------------------------


def _load_existing(
    slug: str,
) -> tuple[BM25Okapi | None, list[str], list[str]]:
    """
    Load an existing BM25 index + metadata for *slug*.

    Returns (bm25 | None, corpus_tokenized, chunk_ids).
    """
    pkl_path = BM25_DIR / f"bm25_{slug}.pkl"
    meta_path = BM25_DIR / f"bm25_{slug}_meta.json"

    if not pkl_path.exists() or not meta_path.exists():
        return None, [], []

    with open(pkl_path, "rb") as f:
        bm25 = pickle.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return bm25, meta.get("corpus_tokenized", []), meta.get("chunk_ids", [])


# ---------------------------------------------------------------------------
# BM25 index orchestration
# ---------------------------------------------------------------------------


def index_bm25(doc_id: str) -> int:
    """Build / update BM25 indexes for all publishers of *doc_id* chunks."""
    if not FINAL_PATH.exists():
        print(f"❌  Error: {FINAL_PATH} not found. Run phase_1_9 first.")
        raise SystemExit(1)

    # Read all chunks for this doc_id, grouped by publisher
    publisher_chunks: dict[str, list[dict]] = defaultdict(list)
    with open(FINAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if record.get("doc_id") == doc_id:
                    pub = record.get("publisher", "unknown")
                    publisher_chunks[pub].append(record)
            except json.JSONDecodeError:
                continue

    if not publisher_chunks:
        print(f"❌  Error: No chunks found for doc_id {doc_id} in {FINAL_PATH}")
        raise SystemExit(1)

    updated_slugs: list[str] = []

    for publisher, chunks in publisher_chunks.items():
        slug = publisher.lower().replace(" ", "_").replace("-", "_")

        # Load existing index (if any)
        existing_bm25, existing_corpus, existing_ids = _load_existing(slug)

        # Build set of existing chunk_ids for dedup
        existing_id_set = set(existing_ids)

        # Filter out already-indexed chunks
        new_chunks = [c for c in chunks if c.get("chunk_id") not in existing_id_set]
        if not new_chunks and existing_bm25 is not None:
            print(f"⏩  No new chunks for publisher '{publisher}' — skipping")
            updated_slugs.append(slug)
            continue

        # Tokenize new chunks
        new_corpus: list[list[str]] = []
        new_ids: list[str] = []
        for c in new_chunks:
            new_corpus.append(_tokenize(c.get("text", "")))
            new_ids.append(c["chunk_id"])

        # Merge with existing data
        merged_corpus = list(existing_corpus) + new_corpus
        merged_ids = list(existing_ids) + new_ids

        if existing_bm25 is not None:
            # Rebuild the index from the merged corpus
            # (rank_bm25 doesn't support incremental updates)
            bm25 = BM25Okapi(merged_corpus)
        else:
            bm25 = BM25Okapi(merged_corpus)

        # Serialize
        pkl_path = BM25_DIR / f"bm25_{slug}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(bm25, f)

        meta_path = BM25_DIR / f"bm25_{slug}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "publisher": publisher,
                "chunk_ids": merged_ids,
            }, f, ensure_ascii=False)

        new_count = len(new_chunks)
        total_count = len(merged_ids)
        updated_slugs.append(slug)
        print(
            f"📄  BM25 '{publisher}' ({slug}): "
            f"{new_count} new + {total_count - new_count} existing = {total_count} docs"
        )

    print(
        f"✅  BM25 index updated for {len(updated_slugs)} publishers: {updated_slugs}"
    )
    return len(updated_slugs)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2.3 — Build/update BM25 indexes per publisher."
    )
    parser.add_argument("--doc_id", type=str, required=True, help="Document UUID")
    args = parser.parse_args()
    index_bm25(args.doc_id)


if __name__ == "__main__":
    main()
