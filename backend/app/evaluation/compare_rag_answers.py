import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.generation.answer import (
    ABSTENTION,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    MODEL_NAME,
    retrieve_context,
    generate_answer,
)
from app.retrieval.hybrid_search import (
    load_chunks,
    dense_search,
    bm25_search,
    exact_match_search,
    reciprocal_rank_fusion,
    tokenize,
)
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

DATASET_PATH = PROJECT_ROOT / "data/evaluation/golden_dataset.json"
OUTPUT_DIR = PROJECT_ROOT / "data/evaluation"
CACHE_PATH = OUTPUT_DIR / "rag_answer_comparison.jsonl"
DETAILS_PATH = OUTPUT_DIR / "rag_answer_comparison_details.csv"
SUMMARY_PATH = OUTPUT_DIR / "rag_answer_comparison_summary.csv"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "enterprise_documents"
METHODS = ("dense", "hybrid")

ABSTENTION_PHRASES = (
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "insufficient information",
)


def token_f1(expected: str, generated: str) -> float:
    """Lexical token-overlap F1; not a measure of semantic correctness."""
    expected_tokens = re.findall(r"\b\w+\b", expected.lower())
    generated_tokens = re.findall(r"\b\w+\b", generated.lower())

    if not expected_tokens or not generated_tokens:
        return 0.0

    expected_counts = Counter(expected_tokens)
    generated_counts = Counter(generated_tokens)
    overlap = sum((expected_counts & generated_counts).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(generated_tokens)
    recall = overlap / len(expected_tokens)

    return 2 * precision * recall / (precision + recall)


def is_abstention(answer: str) -> bool:
    answer = answer.lower()
    return any(phrase in answer for phrase in ABSTENTION_PHRASES)


def load_questions():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    questions = data if isinstance(data, list) else data.get("questions", [])

    if not questions:
        raise ValueError("Golden dataset is empty.")

    return questions


def hybrid_context(question, model, client, chunks, bm25):
    """Retrieve the top five hybrid chunks in the format answer.py expects."""
    dense = dense_search(question, model, client)
    keyword = bm25_search(question, chunks, bm25)
    exact = exact_match_search(question, chunks, bm25)

    ranked = reciprocal_rank_fusion(
        dense,
        keyword,
        question,
        exact_candidates=exact,
    )

    top = ranked[:5]

    contexts = [
        f"[Source: {item['source']}, chunk: {item['chunk_index']}]\n"
        f"{item['text'].strip()}"
        for item in top
        if item.get("text", "").strip()
    ]

    labels = [
        {
            "source": item["source"],
            "chunk_index": int(item["chunk_index"]),
        }
        for item in top
        if item.get("text", "").strip()
    ]

    return contexts, labels


def load_cache():
    """Read saved JSONL results so successful API calls can be reused."""
    cache = {}

    if not CACHE_PATH.exists():
        return cache

    with CACHE_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)
            cache[(row["method"], row["question"])] = row

    return cache


def append_cache(row):
    with CACHE_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_one(item, method, model, client, groq_client, chunks, bm25):
    question = item["question"]
    expected_answer = item.get("expected_answer", "")
    expected_source = item.get("expected_source", "")
    answerable = bool(item["answerable"])

    row = {
        "method": method,
        "question": question,
        "answerable": answerable,
        "expected_source": expected_source,
        "answer": "",
        "abstained": False,
        "token_f1": "",
        "expected_source_cited": False,
        "retrieved_chunks": "[]",
        "status": "ok",
        "error": "",
    }

    try:
        if method == "dense":
            # Use the application's existing retrieval and 0.45 threshold.
            contexts = retrieve_context(question, model, client)

            labels = []
            for context in contexts:
                match = re.match(
                    r"^\[Source: (.*), chunk: (\d+)\]\n",
                    context,
                )
                if match:
                    labels.append({
                        "source": match.group(1),
                        "chunk_index": int(match.group(2)),
                    })

        else:
            # Hybrid RRF scores are not comparable to dense similarity
            # scores, so do not apply the dense threshold to RRF scores.
            contexts, labels = hybrid_context(
                question, model, client, chunks, bm25
            )

        row["retrieved_chunks"] = json.dumps(
            labels, ensure_ascii=False
        )

        if contexts:
            answer = generate_answer(question, contexts, groq_client)
        else:
            answer = ABSTENTION

        row["answer"] = answer
        row["abstained"] = is_abstention(answer)

        if answerable:
            row["token_f1"] = (
                ""
                if row["abstained"]
                else token_f1(expected_answer, answer)
            )
            row["expected_source_cited"] = (
                bool(expected_source) and expected_source in answer
            )

    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def summarize(rows, selected_questions):
    summary_rows = []

    selected_names = {item["question"] for item in selected_questions}

    for method in METHODS:
        method_rows = [
            row for row in rows
            if row["method"] == method
            and row["question"] in selected_names
        ]

        successful = [row for row in method_rows if row["status"] == "ok"]
        answerable = [
            row for row in successful if row["answerable"]
        ]
        unanswerable = [
            row for row in successful if not row["answerable"]
        ]

        non_abstained_answers = [
            row for row in answerable
            if not row["abstained"] and row["token_f1"] != ""
        ]

        summary_rows.append({
            "method": method,
            "questions_in_run": len(selected_questions),
            "successful_evaluations": len(successful),
            "generation_errors": sum(
                row["status"] == "error" for row in method_rows
            ),
            "answerable_questions_evaluated": len(answerable),
            "mean_token_f1_non_abstained": (
                sum(float(row["token_f1"]) for row in non_abstained_answers)
                / len(non_abstained_answers)
                if non_abstained_answers else ""
            ),
            "expected_source_citation_rate": (
                sum(bool(row["expected_source_cited"]) for row in answerable)
                / len(answerable)
                if answerable else ""
            ),
            "unanswerable_questions_evaluated": len(unanswerable),
            "correct_abstention_rate": (
                sum(bool(row["abstained"]) for row in unanswerable)
                / len(unanswerable)
                if unanswerable else ""
            ),
            "false_answer_rate": (
                sum(not bool(row["abstained"]) for row in unanswerable)
                / len(unanswerable)
                if unanswerable else ""
            ),
        })

    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions (for a smoke test).",
    )
    args = parser.parse_args()

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing from the project-root .env file."
        )

    if not MODEL_NAME:
        raise RuntimeError(
            "GROQ_MODEL is missing from the project-root .env file."
        )

    questions = load_questions()

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be greater than zero.")
        questions = questions[:args.limit]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()

    client = QdrantClient(url=QDRANT_URL)
    groq_client = Groq(api_key=GROQ_API_KEY)

    try:
        model = SentenceTransformer(EMBEDDING_MODEL)

        chunks = load_chunks(client)
        if not chunks:
            raise RuntimeError("No indexed chunks found in Qdrant.")

        bm25 = BM25Okapi([
            tokenize(chunk["text"]) for chunk in chunks
        ])

        total = len(questions) * len(METHODS)
        completed = 0

        print(f"Questions selected: {len(questions)}")
        print(f"Methods: {', '.join(METHODS)}")
        print(f"Maximum evaluations in this run: {total}")
        print(f"Saved results cache: {CACHE_PATH}")

        for item in questions:
            for method in METHODS:
                key = (method, item["question"])
                existing = cache.get(key)

                if existing and existing["status"] == "ok":
                    completed += 1
                    print(
                        f"[{completed}/{total}] Reusing saved "
                        f"{method} result: {item['question']}"
                    )
                    continue

                row = evaluate_one(
                    item, method, model, client, groq_client, chunks, bm25
                )

                append_cache(row)
                cache[key] = row
                completed += 1

                if row["status"] == "ok":
                    print(
                        f"[{completed}/{total}] {method.upper()} | "
                        f"abstained={row['abstained']} | "
                        f"question={item['question']}"
                    )
                else:
                    print(
                        f"[{completed}/{total}] ERROR | "
                        f"{method.upper()} | {row['error']}"
                    )

        # Use the latest saved result for every method/question pair.
        current_rows = [
            cache[(method, item["question"])]
            for item in questions
            for method in METHODS
            if (method, item["question"]) in cache
        ]

        details_fields = [
            "method", "question", "answerable", "expected_source",
            "answer", "abstained", "token_f1",
            "expected_source_cited", "retrieved_chunks", "status", "error",
        ]

        with DETAILS_PATH.open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=details_fields)
            writer.writeheader()
            writer.writerows(current_rows)

        summaries = summarize(current_rows, questions)

        with SUMMARY_PATH.open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(summaries[0].keys())
            )
            writer.writeheader()
            writer.writerows(summaries)

        print("\n=== RAG Answer Comparison ===")
        for result in summaries:
            print(f"\nMethod: {result['method'].upper()}")
            for metric in (
                "successful_evaluations",
                "generation_errors",
                "mean_token_f1_non_abstained",
                "expected_source_citation_rate",
                "correct_abstention_rate",
                "false_answer_rate",
            ):
                print(f"  {metric}: {result[metric]}")

        if len(questions) < 40:
            print(
                "\nPARTIAL RUN: These metrics cover only the selected "
                f"{len(questions)} questions."
            )

        print("\nDetailed answers:", DETAILS_PATH)
        print("Summary metrics:", SUMMARY_PATH)
        print("Resumable cache:", CACHE_PATH)

    finally:
        client.close()
        groq_client.close()


if __name__ == "__main__":
    main()
