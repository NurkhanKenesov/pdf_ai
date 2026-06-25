#!/usr/bin/env python3
"""
Phase 2.1 — Embedding

Loads final chunks from data/final_chunks.jsonl, embeds them using
intfloat/multilingual-e5-large via sentence-transformers, and writes
the vectors alongside the original record to data/embeddings.jsonl.

Prefixes each chunk with "passage: " as required by the E5 model.
"""

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


FINAL_PATH = Path("data") / "final_chunks.jsonl"
EMBEDDINGS_PATH = Path("data") / "embeddings.jsonl"

BATCH_SIZE = 32
MODEL_NAME = "intfloat/multilingual-e5-large"


# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("⏳  Loading model intfloat/multilingual-e5-large ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Embedding orchestration
# ---------------------------------------------------------------------------


def embed(doc_id: str) -> int:
    """Embed all final chunks for *doc_id*. Returns number of chunks embedded."""
    if not FINAL_PATH.exists():
        print(f"❌  Error: {FINAL_PATH} not found. Run phase_1_9 first.")
        raise SystemExit(1)

    model = _get_model()

    # Collect chunks for this doc_id (we need batching, so buffer in memory)
    chunks: list[dict] = []
    with open(FINAL_PATH, "r", encoding="utf-8") as f:
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
        print(f"❌  Error: No chunks found for doc_id {doc_id} in {FINAL_PATH}")
        raise SystemExit(1)

    total = len(chunks)
    embedded_count = 0

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as fout:
        for start in range(0, total, BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            texts = ["passage: " + chunk.get("text", "") for chunk in batch]

            embeddings = model.encode(texts, show_progress_bar=False)
            # embeddings is a numpy array of shape (batch_size, 1024)

            for chunk, embedding in zip(batch, embeddings):
                record = dict(chunk)
                record["embedding"] = embedding.tolist()
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            embedded_count += len(batch)
            if embedded_count % (BATCH_SIZE * 10) == 0 or embedded_count == total:
                print(f"⏳  Embedded {embedded_count}/{total} chunks...")

    print(f"✅  Embedded {total} chunks for doc_id {doc_id}")
    return total


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2.1 — Embed chunks using intfloat/multilingual-e5-large."
    )
    parser.add_argument("--doc_id", type=str, required=True, help="Document UUID")
    args = parser.parse_args()
    embed(args.doc_id)


if __name__ == "__main__":
    main()
