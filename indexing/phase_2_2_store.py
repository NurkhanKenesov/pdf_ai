#!/usr/bin/env python3
"""
Phase 2.2 — Vector Store (Qdrant)

Reads embedded chunks from data/embeddings.jsonl and upserts them into a
local Qdrant collection for vector search.

- Qdrant storage: data/qdrant_local (persistent, no Docker)
- Collection:  "ent_knowledge_base"
- Vector size: 1024, distance: Cosine
- Batch upsert: 100 vectors per call
"""

import argparse
import json
import uuid as uuid_mod
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Batch


EMBEDDINGS_PATH = Path("data") / "embeddings.jsonl"
QDRANT_PATH = Path("data") / "qdrant_local"
COLLECTION_NAME = "ent_knowledge_base"
VECTOR_SIZE = 1024
BATCH_SIZE = 100

# Fields to include in the Qdrant payload (everything except "embedding")
PAYLOAD_FIELDS = [
    "chunk_id",
    "doc_id",
    "text",
    "token_count",
    "source_pages",
    "publisher",
    "book_title",
    "language",
    "chapter",
    "formula_present",
    "table_present",
    "has_image",
    "image_path",
]


# ---------------------------------------------------------------------------
# Qdrant client setup
# ---------------------------------------------------------------------------


def _get_client() -> QdrantClient:
    """Return a local persistent Qdrant client."""
    QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    return client


def _ensure_collection(client: QdrantClient) -> None:
    """Create the collection if it does not exist (idempotent)."""
    collections = client.get_collections().collections
    if any(c.name == COLLECTION_NAME for c in collections):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"📦  Created collection '{COLLECTION_NAME}' (size={VECTOR_SIZE}, distance=Cosine)")


# ---------------------------------------------------------------------------
# Store orchestration
# ---------------------------------------------------------------------------


def store(doc_id: str) -> int:
    """Store all embeddings for *doc_id* into Qdrant. Returns vector count."""
    if not EMBEDDINGS_PATH.exists():
        print(f"❌  Error: {EMBEDDINGS_PATH} not found. Run phase_2_1 first.")
        raise SystemExit(1)

    client = _get_client()
    _ensure_collection(client)

    batch_ids: list[int] = []
    batch_vectors: list[list[float]] = []
    batch_payloads: list[dict] = []
    stored = 0
    found_any = False

    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
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

            found_any = True

            # UUID string → integer for Qdrant
            point_id = int(uuid_mod.UUID(record["chunk_id"]))
            vector = record["embedding"]

            payload = {field: record.get(field) for field in PAYLOAD_FIELDS}
            # Convert source_pages to list if needed for JSON serialization
            if isinstance(payload.get("source_pages"), list):
                payload["source_pages"] = [int(p) for p in payload["source_pages"]]

            batch_ids.append(point_id)
            batch_vectors.append(vector)
            batch_payloads.append(payload)

            # Flush when batch is full
            if len(batch_ids) >= BATCH_SIZE:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=Batch(
                        ids=batch_ids,
                        vectors=batch_vectors,
                        payloads=batch_payloads,
                    ),
                )
                stored += len(batch_ids)
                batch_ids, batch_vectors, batch_payloads = [], [], []

    if not found_any:
        print(f"❌  Error: No embeddings found for doc_id {doc_id} in {EMBEDDINGS_PATH}")
        raise SystemExit(1)

    # Flush remainder
    if batch_ids:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=Batch(
                ids=batch_ids,
                vectors=batch_vectors,
                payloads=batch_payloads,
            ),
        )
        stored += len(batch_ids)

    print(
        f"✅  Stored {stored} vectors in Qdrant collection "
        f'"{COLLECTION_NAME}"'
    )
    return stored


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2.2 — Store embeddings in local Qdrant."
    )
    parser.add_argument("--doc_id", type=str, required=True, help="Document UUID")
    args = parser.parse_args()
    store(args.doc_id)


if __name__ == "__main__":
    main()
