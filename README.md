# pdf_ai — ENT Textbook RAG Pipeline

A 4-phase RAG (Retrieval-Augmented Generation) pipeline for Kazakh and Russian educational textbooks. Built for ENT (Единое Национальное Тестирование / Ұлттық Бірыңғай Тестілеу) preparation materials.

## Pipeline Overview

```
PDF → [Ingestion] → [Indexing] → [Retrieval] → [Generation] → Answer
```

### Phase 1 — Ingestion (phases 1.1–1.9)
Extracts text from scanned PDF textbooks using OCR (Tesseract with `kaz+rus` / `rus+kaz`) with a PyMuPDF fallback chain. Splits content into semantic chunks (~400 tokens each) and runs quality assurance.

### Phase 2 — Indexing (phases 2.1–2.4)
- **Embedding:** `intfloat/multilingual-e5-large` (1024d vectors)
- **Vector Store:** Local Qdrant (`ent_knowledge_base` collection, cosine distance)
- **Sparse Index:** Per-publisher BM25 indexes via `rank-bm25`

### Phase 3 — Retrieval (phase 3)
Multi-stage search pipeline:
1. Language detection + query classification + synonym expansion
2. Parallel dense (Qdrant) + sparse (BM25) search
3. Reciprocal Rank Fusion (RRF, k=60)
4. Cross-encoder reranking (`BAAI/bge-reranker-base`, optional fallback)

### Phase 4 — Generation (phase 4)
- Token budget management (max 8000 tokens, truncates lowest-scoring chunks)
- Context assembly with source citations
- LLM integration (OpenAI-compatible — xAI Grok or any OpenAI-compatible API)
- Citation verification and quality assurance checks

## Setup

```bash
# Clone
git clone https://github.com/NurkhanKenesov/pdf_ai.git
cd pdf_ai

# Install dependencies
pip install sentence-transformers qdrant-client rank-bm25 pytesseract pdfplumber PyMuPDF pillow openai

# Install Tesseract with Kazakh + Russian language packs
# macOS:
brew install tesseract tesseract-lang
# Ubuntu:
sudo apt install tesseract-ocr tesseract-ocr-kaz tesseract-ocr-rus
```

## Usage

```bash
# 1. Ingest a PDF
python3 ingestion/phase_1_1_register.py /path/to/textbook.pdf
python3 ingestion/phase_1_2_detect_type.py /path/to/textbook.pdf
python3 ingestion/phase_1_3_detect_content.py /path/to/textbook.pdf
python3 ingestion/phase_1_4_extract.py /path/to/textbook.pdf
python3 ingestion/phase_1_5_normalize.py --doc_id <UUID>
python3 ingestion/phase_1_6_chunk.py --doc_id <UUID>
python3 ingestion/phase_1_7_enrich.py --doc_id <UUID> --publisher "Мектеп"
python3 ingestion/phase_1_8_qa.py --doc_id <UUID>
python3 ingestion/phase_1_9_export.py --doc_id <UUID>

# 2. Index
python3 indexing/phase_2_1_embed.py --doc_id <UUID>
python3 indexing/phase_2_2_store.py --doc_id <UUID>
python3 indexing/phase_2_3_bm25.py --doc_id <UUID>
python3 indexing/phase_2_4_verify.py

# 3. Retrieve
python3 retrieval/phase_3_retrieve.py --query "Түрік қағанаты қашан құрылды" --top_n 3

# 4. Generate
# Set XAI_API_KEY and XAI_BASE_URL environment variables
python3 generation/phase_4_generate.py
```

## Requirements

- Python 3.11+
- Tesseract OCR with Kazakh and Russian language packs
- macOS or Linux (tested on macOS)

## Project Structure

```
├── ingestion/          # Phase 1: PDF ingestion pipeline (9 steps)
├── indexing/           # Phase 2: Embedding + Qdrant + BM25 (4 steps)
├── retrieval/          # Phase 3: Hybrid search + reranking (1 step)
├── generation/         # Phase 4: LLM generation with citations (1 step)
└── data/               # Pipeline outputs (gitignored)
```
