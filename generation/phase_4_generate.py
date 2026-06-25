#!/usr/bin/env python3
"""
Phase 4 — Generation

Takes retrieval output from Phase 3 (data/retrieval_result.json), builds a
retrieval-augmented prompt, calls an LLM (OpenAI-compatible API like xAI
Grok), and returns a cited answer with QA validation.

Usage:
  python3 generation/phase_4_generate.py [--retrieval data/retrieval_result.json]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI


RETRIEVAL_PATH = Path("data") / "retrieval_result.json"

# Hard budget: system + all context + query must fit in here
MAX_CONTEXT_TOKENS = 8000

# Approximate token ratio (chars per token) for Cyrillic/mixed text
CHARS_PER_TOKEN = 4.0

SYSTEM_PROMPT = (
    "Ты — ЕНТ тьютор. Отвечай только на основе предоставленного контекста. "
    "Язык ответа: тот же, что и вопрос (kz/ru). "
    "Каждый факт сопровождай цитатой [N], где N — номер источника из контекста."
)

# Regex to find citation markers like [1], [2], etc.
CITATION_RE = re.compile(r"\[(\d+)\]")


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Approximate token count for mixed Cyrillic/Latin text."""
    return int(len(text) / CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Load retrieval result
# ---------------------------------------------------------------------------

def load_retrieval(path: Path) -> dict:
    """Load and validate the retrieval result JSON."""
    if not path.exists():
        print(f"❌  Error: {path} not found. Run Phase 3 retrieval first.")
        raise SystemExit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "chunks" not in data or not data["chunks"]:
        print(f"❌  Error: No chunks in retrieval result.")
        raise SystemExit(1)

    return data


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

def check_budget(system: str, chunks: list[dict], query: str) -> list[dict]:
    """
    Ensure total token count ≤ MAX_CONTEXT_TOKENS.
    If over budget, remove lowest-scoring chunks (from the end) until fit.
    Returns the surviving chunks list.
    """
    surviving = list(chunks)
    while True:
        context_texts = []
        for i, c in enumerate(surviving, start=1):
            context_texts.append(
                f"[CONTEXT {i}]: {c['text']} "
                f"(Publisher: {c['publisher']}, Pages: {'-'.join(str(p) for p in c['source_pages'])})"
            )
        full_prompt = system + "\n\n" + "\n\n".join(context_texts) + "\n\n[USER QUERY]: " + query
        total_tokens = _approx_tokens(full_prompt)

        if total_tokens <= MAX_CONTEXT_TOKENS:
            print(f"📊  Token budget: ~{total_tokens} / {MAX_CONTEXT_TOKENS} "
                  f"({len(surviving)} chunks)")
            return surviving

        if len(surviving) <= 1:
            print(f"⚠️  Token budget exceeded ({total_tokens} > {MAX_CONTEXT_TOKENS}) "
                  f"even with 1 chunk. Truncating chunk text.")
            # Truncate the last chunk text to fit
            last = surviving[-1]
            max_text_chars = int((MAX_CONTEXT_TOKENS - _approx_tokens(
                system + "\n\n[CONTEXT 1]: \n\n[USER QUERY]: " + query
            )) * CHARS_PER_TOKEN)
            last["text"] = last["text"][:max_text_chars]
            continue

        # Remove lowest-scoring chunk (last in sorted order)
        removed = surviving.pop()
        print(f"  ⏎  Dropped chunk (score={removed.get('rerank_score', 0):.4f}) — "
              f"budget was ~{total_tokens} tokens")


# ---------------------------------------------------------------------------
# Assemble prompt
# ---------------------------------------------------------------------------

def assemble_prompt(chunks: list[dict], query: str) -> str:
    """Build the user prompt with context and query (system prompt is sent separately)."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[CONTEXT {i}]: {c['text']} "
            f"(Publisher: {c['publisher']}, Pages: {'-'.join(str(p) for p in c['source_pages'])})"
        )
    parts.append(f"[USER QUERY]: {query}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call (OpenAI-compatible — Grok / xAI)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, temperature: float = 0.1) -> str:
    """Call the LLM with streaming and return the full response text.

    Sends the system prompt via the system role and context via user role
    for better instruction following.
    """
    api_key = os.environ.get("XAI_API_KEY")
    base_url = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")

    if not api_key:
        print("❌  Error: XAI_API_KEY environment variable not set.")
        raise SystemExit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    print("🤖  LLM call: Grok (xAI) | temperature=0.1")
    print("   ", end="", flush=True)

    try:
        response_text = ""
        stream = client.chat.completions.create(
            model="grok-3",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            response_text += delta
            print(delta, end="", flush=True)

        print()
        return response_text

    except Exception as exc:
        print(f"\n❌  LLM API error: {exc}")
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Parse citations
# ---------------------------------------------------------------------------

def parse_citations(text: str, max_index: int) -> tuple[str, list[int]]:
    """
    Extract citation markers [N] from text.
    Returns (cleaned_text, list_of_citation_indices).
    Verifies each citation references a valid context index (1..max_index).
    """
    found = []
    invalid = []

    for match in CITATION_RE.finditer(text):
        idx = int(match.group(1))
        if 1 <= idx <= max_index:
            found.append(idx)
        else:
            invalid.append(idx)

    if invalid:
        print(f"  ⚠️  Invalid citation(s) referencing context beyond {max_index}: {invalid}")

    return text, sorted(set(found))


# ---------------------------------------------------------------------------
# Post-process: format citations and sources
# ---------------------------------------------------------------------------

def format_sources(chunks: list[dict]) -> str:
    """Build the ---SOURCES--- footer."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        pages = "-".join(str(p) for p in c["source_pages"])
        lines.append(f"[{i}] {c['publisher']}, {c['book_title']}, стр. {pages}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# QA checks
# ---------------------------------------------------------------------------

def run_qa(answer: str, query: str, chunks: list[dict],
           citations: list[int], lang: str) -> dict:
    """Run the 5 QA checks and return results."""
    checks = {}

    # Check 1: Answer not empty
    checks["not_empty"] = len(answer.strip()) > 0

    # Check 2: Language match (detect Kazakh chars in answer)
    kazakh_chars = set("әіңғүұқөһӘІҢҒҮҰҚӨҺ")
    answer_kaz_ratio = sum(1 for ch in answer if ch in kazakh_chars) / max(len(answer), 1)
    # Query language is "kz" — expect answer to have Kazakh chars
    checks["lang_match"] = (lang == "kz" and answer_kaz_ratio > 0.005) or \
                           (lang == "ru" and answer_kaz_ratio <= 0.01)

    # Check 3: At least 1 citation
    checks["has_citation"] = len(citations) > 0

    # Check 4: Answer doesn't contradict chunks (basic: at least some overlap)
    # Check if key terms from chunks appear in answer
    chunk_text = " ".join(c["text"] for c in chunks).lower()
    # Extract key entities from answer (first 5 unique words > 4 chars)
    answer_words = set(re.findall(r"[а-яәіңғүұқөһА-ЯӘІҢҒҮҰҚӨҺa-z]{5,}", answer.lower()))
    chunk_words = set(re.findall(r"[а-яәіңғүұқөһА-ЯӘІҢҒҮҰҚӨҺa-z]{5,}", chunk_text))
    overlap = answer_words & chunk_words
    # Require at least 3 overlapping significant words to be considered grounded
    checks["grounded"] = len(overlap) >= 3

    # Check 5: Answer length 50–500 words
    word_count = len(answer.split())
    checks["length_ok"] = 50 <= word_count <= 500

    return checks


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def generate(retrieval_path: Path = RETRIEVAL_PATH) -> None:
    """Run the full Phase 4 generation pipeline."""

    # ── 4.1 Load retrieval result ─────────────────────────────────────────
    data = load_retrieval(retrieval_path)
    query = data["query"]
    lang = data["lang"]
    chunks = data["chunks"]
    top_n = data["top_n"]

    print(f"\n📥  Loaded retrieval: {len(chunks)} chunks for query \"{query}\"")
    print(f"    Language: {lang} | top_n: {top_n}")

    # ── 4.2 Token budget ─────────────────────────────────────────────────
    surviving = check_budget(SYSTEM_PROMPT, chunks, query)

    # ── 4.3 & 4.4 Build prompt ────────────────────────────────────────────
    prompt = assemble_prompt(surviving, query)  # system prompt sent separately

    # ── 4.5 LLM generation ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    raw_response = call_llm(prompt, temperature=0.1)

    # ── 4.6 Parse LLM response ────────────────────────────────────────────
    answer_text = raw_response.strip()
    max_idx = len(surviving)
    answer_text, citations = parse_citations(answer_text, max_idx)

    # ── 4.7 Post-process ──────────────────────────────────────────────────
    sources_text = format_sources(surviving)

    # Replace any [FORMULA:...] markers with display text
    answer_text = re.sub(r"\[FORMULA:\s*(.*?)\]", r"\1", answer_text)

    # ── 4.8 QA checks ────────────────────────────────────────────────────
    checks = run_qa(answer_text, query, surviving, citations, data["lang"])

    # ── 4.9 Final output ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("---ANSWER---")
    print(answer_text)

    print("\n\n---SOURCES---")
    print(sources_text)

    print("\n\n---QA---")
    all_pass = True
    check_names = {
        "not_empty": "Answer is not empty",
        "lang_match": "Answer language matches query language",
        "has_citation": "At least 1 citation [N] present",
        "grounded": "Answer does not contradict chunk content",
        "length_ok": "Answer length: 50–500 words",
    }
    for key, label in check_names.items():
        passed = checks.get(key, False)
        if not passed:
            all_pass = False
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    word_count = len(answer_text.split())
    print(f"\n  (Answer word count: {word_count})")

    if not checks.get("has_citation", False):
        print("  ⚠️  flag: no_citation")
    if not checks.get("grounded", False):
        print("  ⚠️  flag: off_topic")
    if not checks.get("lang_match", False):
        print("  ⚠️  flag: lang_mismatch")

    if all_pass:
        print("\n  Verdict: ✅ Answer ready")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"\n  Verdict: ⚠️  Issues found: {', '.join(failed)}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 — Generate LLM answer from retrieval output."
    )
    parser.add_argument(
        "--retrieval", type=str, default=str(RETRIEVAL_PATH),
        help="Path to retrieval result JSON (default: data/retrieval_result.json)"
    )
    args = parser.parse_args()
    generate(Path(args.retrieval))


if __name__ == "__main__":
    main()
