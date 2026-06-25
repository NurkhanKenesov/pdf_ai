#!/usr/bin/env python3
"""
Phase 2.4 — Verification

Full end-to-end verification of the Phase 2 indexes:

1. Qdrant: count vectors in "ent_knowledge_base"
2. Qdrant: run a test vector query
3. BM25: load all bm25_*.pkl indexes and run a test keyword query
4. Cross-check: count lines in final_chunks.jsonl vs Qdrant vector count

Output: printed verification report.
"""

import json
import pickle
import re as re_mod
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


FINAL_PATH = Path("data") / "final_chunks.jsonl"
QDRANT_PATH = Path("data") / "qdrant_local"
COLLECTION_NAME = "ent_knowledge_base"
MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 3

# Test queries
VECTOR_QUERY = "Түрік қағанаты қашан құрылды"
BM25_QUERY_TOKENS = ["түрік", "қағанаты"]

PUNCT_RE = re_mod.compile(r"[^\w\s]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return PUNCT_RE.sub("", text.lower()).split()


def _count_lines_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ---------------------------------------------------------------------------
# 1. Qdrant verification
# ---------------------------------------------------------------------------


def _search_qdrant(client: QdrantClient, embedding: list[float]) -> list[dict]:
    """Run a test vector search and return top-K results."""
    qdrant_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=TOP_K,
    ).points
    return [
        {
            "chunk_id": result.payload.get("chunk_id", "?"),
            "publisher": result.payload.get("publisher", "?"),
            "score": round(result.score, 4),
            "text_preview": (result.payload.get("text", "") or "")[:80],
        }
        for result in qdrant_results
    ]


# ---------------------------------------------------------------------------
# 2. BM25 verification
# ---------------------------------------------------------------------------


def _find_bm25_indexes() -> list[tuple[str, str, Path, Path]]:
    """
    Find all BM25 index/metadata files.

    Returns list of (publisher_name, slug, pkl_path, meta_path).
    """
    results: list[tuple[str, str, Path, Path]] = []
    for pkl_path in sorted(Path("data").glob("bm25_*.pkl")):
        slug = pkl_path.stem.replace("bm25_", "")
        meta_path = Path("data") / f"bm25_{slug}_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        results.append((meta.get("publisher", slug), slug, pkl_path, meta_path))
    return results


def _search_bm25(pkl_path: Path, tokens: list[str]) -> list[dict]:
    """Load a BM25 index and search with *tokens*. Returns top-K results."""
    with open(pkl_path, "rb") as f:
        bm25 = pickle.load(f)

    scores = bm25.get_scores(tokens)
    # Get top-K indices
    indexed = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:TOP_K]

    meta_path = pkl_path.parent / f"{pkl_path.stem}_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chunk_ids = meta.get("chunk_ids", [])
    # Read tokenized corpus from the BM25 object itself (avoids duplication)
    corpus = bm25.corpus if hasattr(bm25, "corpus") else []

    results: list[dict] = []
    for idx, score in indexed:
        if score == 0:
            continue
        results.append({
            "chunk_id": chunk_ids[idx] if idx < len(chunk_ids) else "?",
            "score": round(float(score), 4),
            "text_preview": " ".join(corpus[idx][:15]) if idx < len(corpus) else "?",
        })
    return results


# ---------------------------------------------------------------------------
# Verification orchestration
# ---------------------------------------------------------------------------


def verify() -> int:
    """Run all verification checks and print a full report."""
    print("=== Phase 2 Verification ===\n")

    issues: list[str] = []

    # ── 1. Qdrant ─────────────────────────────────────────────────────────
    print("── Qdrant ──")
    qdrant_exists = QDRANT_PATH.exists()
    if not qdrant_exists:
        print("⚠️  Qdrant storage not found at data/qdrant_local")
        issues.append("Qdrant storage missing")
        qdrant_count = 0
        qdrant_results = []
    else:
        client = QdrantClient(path=str(QDRANT_PATH))
        qdrant_count = client.count(collection_name=COLLECTION_NAME).count
        print(f"  Vectors in 'ent_knowledge_base': {qdrant_count}")
        if qdrant_count < 16:
            print(f"  ⚠ Expected 16, got {qdrant_count} — re-run phase_2_2_store.py")

        # Test query
        if qdrant_count > 0:
            print(f"\n  Test vector query: \"{VECTOR_QUERY}\"")
            model = SentenceTransformer(MODEL_NAME)
            embedding = model.encode("query: " + VECTOR_QUERY)
            qdrant_results = _search_qdrant(client, embedding.tolist())
            for r in qdrant_results:
                print(
                    f"    [{r['publisher']}] score={r['score']:.4f}  "
                    f"{r['chunk_id'][:8]}...  "
                    f"{r['text_preview']}"
                )
        else:
            qdrant_results = []
            print("  ⚠️  No vectors to query")

    print()

    # ── 2. BM25 ───────────────────────────────────────────────────────────
    print("── BM25 Indexes ──")
    bm25_indexes = _find_bm25_indexes()
    if not bm25_indexes:
        print("  (none found)")
    else:
        bm25_results: dict[str, list[dict]] = {}
        for pub_name, slug, pkl_path, _ in bm25_indexes:
            print(f"  Publisher: {pub_name} ({slug})")
            results = _search_bm25(pkl_path, BM25_QUERY_TOKENS)
            bm25_results[slug] = results
            for r in results:
                print(
                    f"    score={r['score']:.4f}  "
                    f"{r['chunk_id'][:8]}...  "
                    f"{r['text_preview']}"
                )

    print()

    # ── 3. Cross-check ────────────────────────────────────────────────────
    print("── Cross-check ──")
    final_count = _count_lines_jsonl(FINAL_PATH)
    print(f"  Lines in final_chunks.jsonl:   {final_count}")
    print(f"  Vectors in Qdrant:             {qdrant_count}")
    if final_count != qdrant_count:
        issues.append(
            f"Qdrant vector count ({qdrant_count}) != "
            f"final_chunks lines ({final_count})"
        )

    if issues:
        print(f"\n  ⚠️  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  ✅  All counts match")

    print()
    status = "⚠️  MISMATCH" if issues else "✅ OK"
    print(f"Status: {status}")

    return len(issues)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    verify()


if __name__ == "__main__":
    main()
