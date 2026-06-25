#!/usr/bin/env python3
"""
Phase 3 — Retrieval

Core search module: takes a user query and returns top-N chunks ready for LLM.
Combines dense (Qdrant) + sparse (BM25) search with RRF fusion and
optional cross-encoder reranking (BAAI/bge-reranker-base).

CLI usage:
  python3 retrieval/phase_3_retrieve.py --query "Түрік қағанаты қашан құрылды" --top_n 3
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
QDRANT_PATH = DATA_DIR / "qdrant_local"
COLLECTION_NAME = "ent_knowledge_base"
FINAL_CHUNKS_PATH = DATA_DIR / "final_chunks.jsonl"
OUTPUT_PATH = DATA_DIR / "retrieval_result.json"

MODEL_NAME = "intfloat/multilingual-e5-large"
RERANKER_NAME = "BAAI/bge-reranker-base"

# ── Search constants ───────────────────────────────────────────────────────
TOP_K_VECTOR = 20
TOP_K_BM25 = 20
TOP_K_RRF = 20
RRF_K = 60

# ── Text processing ────────────────────────────────────────────────────────
PUNCT_RE = re.compile(r"[^\w\s]")
KAZAKH_CHARS: set[str] = set("әіңғүұқөһ")

SYNONYMS: dict[str, list[str]] = {
    "қағанат": ["мемлекет", "империя"],
    "хан": ["қаған", "билеуші"],
    "государство": ["мемлекет", "держава"],
    "год": ["жыл", "жылы"],
}

QUESTION_WORDS: set[str] = {"кто", "что", "когда", "где", "қашан", "кім", "не", "қай"}
EXPLAIN_WORDS: set[str] = {"почему", "как", "объясни", "түсіндір", "неліктен", "қалай"}
TASK_WORDS: set[str] = {"реши", "вычисли", "найди", "есептеу", "тап"}


# ── Helpers ────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return PUNCT_RE.sub("", text.lower()).split()


def _detect_lang(query: str) -> str:
    """Detect Kazakh if >10% of characters are Kazakh-specific."""
    total = len(query.strip())
    if total == 0:
        return "ru"
    kz_count = sum(1 for c in query if c in KAZAKH_CHARS)
    return "kz" if (kz_count / total) > 0.1 else "ru"


def _classify_query(query: str) -> str:
    """Classify query as fact / explain / task by question-word heuristics."""
    words = set(_tokenize(query))
    if words & EXPLAIN_WORDS:
        return "explain"
    if words & TASK_WORDS:
        return "task"
    if words & QUESTION_WORDS:
        return "fact"
    return "fact"


def _expand_query(query: str) -> str:
    """
    Normalise (lowercase, strip punctuation) and append synonyms from the
    hardcoded dictionary for any matching substrings.
    """
    normalized = PUNCT_RE.sub("", query.lower().strip())
    extra_terms: list[str] = []
    for word, syns in SYNONYMS.items():
        if word in normalized:
            extra_terms.extend(syns)
    if extra_terms:
        return normalized + " " + " ".join(extra_terms)
    return normalized


def _find_bm25_indexes(publisher_slug: Optional[str] = None) -> list[dict]:
    """
    Discover BM25 index files, optionally filtering by publisher slug.

    Returns a list of dicts with keys:
        slug, publisher_name, pkl_path, meta_path.
    """
    results: list[dict] = []

    def _add(slug: str) -> None:
        pkl = DATA_DIR / f"bm25_{slug}.pkl"
        meta = DATA_DIR / f"bm25_{slug}_meta.json"
        if not pkl.exists():
            return
        pub_name = "unknown"
        if meta.exists():
            with open(meta, "r", encoding="utf-8") as fh:
                pub_name = json.load(fh).get("publisher", "unknown")
        results.append({
            "slug": slug,
            "publisher_name": pub_name,
            "pkl_path": pkl,
            "meta_path": meta,
        })

    if publisher_slug:
        _add(publisher_slug)
    else:
        for pkl_path in sorted(DATA_DIR.glob("bm25_*.pkl")):
            slug = pkl_path.stem.replace("bm25_", "")
            _add(slug)
    return results


def _lookup_chunk(chunk_id: str) -> Optional[dict]:
    """Scan final_chunks.jsonl for a chunk by chunk_id (fast for small docs)."""
    if not FINAL_CHUNKS_PATH.exists():
        return None
    with open(FINAL_CHUNKS_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
                if rec.get("chunk_id") == chunk_id:
                    return rec
            except json.JSONDecodeError:
                continue
    return None


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 — Retrieve top-N chunks for a user query."
    )
    parser.add_argument("--query", type=str, required=True, help="Search text")
    parser.add_argument(
        "--publisher", type=str, default=None,
        help="Filter by publisher slug (e.g. мектеп). Default: search all."
    )
    parser.add_argument(
        "--top_n", type=int, default=5,
        help="Number of chunks to return (default: 5)"
    )
    parser.add_argument(
        "--lang", type=str, default="auto",
        choices=["ru", "kz", "auto"],
        help="Language override (default: auto-detect)"
    )
    return parser.parse_args()


# ── Pipeline ───────────────────────────────────────────────────────────────


def retrieve(
    query: str,
    publisher_slug: Optional[str] = None,
    top_n: int = 5,
    lang_override: str = "auto",
) -> dict:
    """
    Full retrieval pipeline:

    1. Language detection & query classification
    2. Query expansion
    3. Embedding (multilingual-e5-large)
    4. Parallel vector (Qdrant) + sparse (BM25) search
    5. Reciprocal Rank Fusion
    6. Cross-encoder reranking (fallback to RRF order)
    7. Context assembly & JSON output
    """
    # ── 3.2 Language detection & query classification ──────────────────────
    lang = _detect_lang(query) if lang_override == "auto" else lang_override
    query_type = _classify_query(query)
    print(f"🔍 Query: \"{query}\" | lang: {lang} | type: {query_type}")

    # ── 3.3 Query expansion ────────────────────────────────────────────────
    expanded = _expand_query(query)
    print(f"📝 Expanded: \"{expanded}\"")

    # ── 3.4 Query embedding ────────────────────────────────────────────────
    print("⏳  Loading embedding model ...")
    model = SentenceTransformer(MODEL_NAME)
    query_vector: list[float] = model.encode("query: " + expanded).tolist()

    # ── 3.5 Parallel search ────────────────────────────────────────────────
    # ── Vector (dense) ─────────────────────────────────────────────────────
    print("🔎  Vector search (Qdrant) ...")
    client = QdrantClient(path=str(QDRANT_PATH))

    qdrant_filter: Optional[Filter] = None
    if publisher_slug:
        indexes = _find_bm25_indexes(publisher_slug)
        if indexes:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="publisher",
                        match=MatchValue(value=indexes[0]["publisher_name"]),
                    )
                ]
            )

    qdrant_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K_VECTOR,
        query_filter=qdrant_filter,
    )

    vector_results: list[dict] = []
    payload_map: dict[str, dict] = {}
    for idx, point in enumerate(qdrant_result.points):
        payload = point.payload or {}
        chunk_id = payload.get("chunk_id", f"v{idx}")
        vector_results.append({"chunk_id": chunk_id, "rank": idx + 1})
        payload_map[chunk_id] = payload

    print(f"  → {len(vector_results)} results from Qdrant")

    # ── Sparse (BM25) ──────────────────────────────────────────────────────
    print("🔎  BM25 search ...")
    bm25_indexes = _find_bm25_indexes(publisher_slug)
    bm25_tokens = _tokenize(expanded)

    bm25_results: list[dict] = []
    for info in bm25_indexes:
        with open(info["pkl_path"], "rb") as fh:
            bm25 = pickle.load(fh)
        with open(info["meta_path"], "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        chunk_ids: list[str] = meta.get("chunk_ids", [])
        scores = bm25.get_scores(bm25_tokens)

        top_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:TOP_K_BM25]

        slug_results: list[dict] = []
        for bm25_idx, score in top_indices:
            if score == 0:
                continue
            if bm25_idx < len(chunk_ids):
                slug_results.append({
                    "chunk_id": chunk_ids[bm25_idx],
                    "rank": len(slug_results) + 1,
                })

        bm25_results.extend(slug_results)
        print(f"  → {info['slug']}: {len(slug_results)} BM25 results")

    if not bm25_indexes:
        print("  → No BM25 indexes found")

    # ── 3.6 Reciprocal Rank Fusion ─────────────────────────────────────────
    rrf: dict[str, float] = {}
    for vr in vector_results:
        cid = vr["chunk_id"]
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + vr["rank"])
    for br in bm25_results:
        cid = br["chunk_id"]
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + br["rank"])

    rrf_ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:TOP_K_RRF]
    print(f"🔀 RRF candidates: {len(rrf_ranked)} unique chunks")

    # ── 3.7 Reranking (Cross-encoder) ──────────────────────────────────────
    reranker_available = True
    try:
        from sentence_transformers import CrossEncoder as _CrossEncoder
        cross_encoder = _CrossEncoder(RERANKER_NAME)
    except Exception:
        reranker_available = False
        cross_encoder = None

    if reranker_available and rrf_ranked:
        pairs: list[tuple[str, str]] = []
        for cid, _ in rrf_ranked:
            text = ""
            payload = payload_map.get(cid)
            if payload and payload.get("text"):
                text = payload["text"]
            else:
                chunk_data = _lookup_chunk(cid)
                if chunk_data and chunk_data.get("text"):
                    text = chunk_data["text"]
            pairs.append((query, text))

        rerank_scores: list[float] = cross_encoder.predict(pairs).tolist()

        combined = [
            (cid, rrf_score, rerank_score)
            for (cid, rrf_score), rerank_score in zip(rrf_ranked, rerank_scores)
        ]
        combined.sort(key=lambda x: x[2], reverse=True)
        top_chunks = combined[:top_n]
        print(f"🎯 Reranked: top {len(top_chunks)} selected")
    else:
        if not reranker_available:
            print("⚠️  Reranker unavailable — using RRF order")
        top_chunks = [
            (cid, rrf_score, rrf_score) for cid, rrf_score in rrf_ranked[:top_n]
        ]

    # ── 3.8 Context assembly ───────────────────────────────────────────────
    chunks_out: list[dict] = []
    for rank, (cid, rrf_score, rerank_score) in enumerate(top_chunks, start=1):
        payload = payload_map.get(cid)
        if not payload:
            payload = _lookup_chunk(cid) or {}

        text = payload.get("text", "")
        source_pages_raw = payload.get("source_pages", [])
        if isinstance(source_pages_raw, list):
            source_pages = [int(p) for p in source_pages_raw]
        else:
            source_pages = [int(source_pages_raw)] if source_pages_raw else []

        block = {
            "chunk_id": cid,
            "rank": rank,
            "text": text,
            "publisher": payload.get("publisher", ""),
            "book_title": payload.get("book_title", ""),
            "source_pages": source_pages,
            "language": payload.get("language", lang),
            "formula_present": bool(payload.get("formula_present", False)),
            "table_present": bool(payload.get("table_present", False)),
            "rrf_score": round(rrf_score, 6),
            "rerank_score": round(rerank_score, 6),
        }
        chunks_out.append(block)

        # Pretty print preview
        pub = block["publisher"]
        if source_pages:
            if len(source_pages) == 1:
                pages_str = f"стр.{source_pages[0]}"
            else:
                pages_str = f"стр.{source_pages[0]}-{source_pages[-1]}"
        else:
            pages_str = ""
        preview = text[:120].replace("\n", " ").strip()
        print(f"  [{rank}] {pub} · {pages_str} · score: {rerank_score:.4f}")
        print(f'      "{preview}..."')

    # ── 3.9 Output ─────────────────────────────────────────────────────────
    result = {
        "query": query,
        "expanded_query": expanded,
        "lang": lang,
        "query_type": query_type,
        "top_n": top_n,
        "chunks": chunks_out,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"\n✅ Retrieval complete: {len(chunks_out)} chunks ready for generation")
    return result


def main() -> None:
    args = parse_args()
    retrieve(
        query=args.query,
        publisher_slug=args.publisher,
        top_n=args.top_n,
        lang_override=args.lang,
    )


if __name__ == "__main__":
    main()
