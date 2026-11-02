
from pathlib import Path
import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from app.retrieval.hybrid_search import (
    load_chunks,
    dense_search,
    bm25_search,
    reciprocal_rank_fusion,
    exact_match_search,
    tokenize,
)

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "enterprise_documents"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 5
CANDIDATE_K = 30

# Relevant labels verified against the actual chunk text.
EVALUATION_CASES = [
    {
        "question": (
            "How many annual leave days do employees "
            "with less than 3 years of service receive?"
        ),
        "relevant": {("employee_handbook (1).pdf", 2)},
    },
    {
        "question": "Who approves annual leave requests?",
        "relevant": {("employee_handbook (1).pdf", 3)},
    },
    {
        "question": "How should employees request annual leave?",
        "relevant": {("employee_handbook (1).pdf", 3)},
    },
    {
        "question": "How many paid sick leave days are allowed per year?",
        "relevant": {("employee_handbook (1).pdf", 4)},
    },
    {
        "question": "What is the IT Security hotline extension?",
        "relevant": {
            ("it_security_policy.pdf", 8),
            ("it_security_policy.pdf", 13),
        },
    },
]


def label(result):
    return (result["source"], result["chunk_index"])


def recall_at_k(results, relevant, k=TOP_K):
    retrieved = {label(item) for item in results[:k]}
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank_at_k(results, relevant, k=TOP_K):
    for rank, item in enumerate(results[:k], start=1):
        if label(item) in relevant:
            return 1 / rank
    return 0.0


def main():
    client = QdrantClient(url=QDRANT_URL)

    print("Loading Qdrant chunks...")
    chunks = load_chunks(client)
    print(f"Loaded {len(chunks)} chunks.")

    if not chunks:
        raise RuntimeError("No chunks found in Qdrant.")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    tokenized_corpus = [tokenize(item["text"]) for item in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    totals = {
        "Dense": {"recall": 0.0, "mrr": 0.0},
        "BM25": {"recall": 0.0, "mrr": 0.0},
        "Hybrid": {"recall": 0.0, "mrr": 0.0},
    }

    for case in EVALUATION_CASES:
        question = case["question"]
        relevant = case["relevant"]

        dense_results = dense_search(
            question,
            model,
            client,
        )

        bm25_results = bm25_search(
            question,
            chunks,
            bm25,
        )

        exact_results = exact_match_search(
            question,
            chunks,
            bm25,
        )

        hybrid_results = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            question,
            exact_results,
        )

        methods = {
            "Dense": dense_results,
            "BM25": bm25_results,
            "Hybrid": hybrid_results,
        }

        print(f"\nQuestion: {question}")
        print("Relevant chunks:", sorted(relevant))

        for method, results in methods.items():
            r5 = recall_at_k(results, relevant)
            mrr5 = reciprocal_rank_at_k(results, relevant)

            totals[method]["recall"] += r5
            totals[method]["mrr"] += mrr5

            print(
                f"{method:7} | "
                f"Recall@{TOP_K}={r5:.2f} | "
                f"MRR@{TOP_K}={mrr5:.2f} | "
                f"Top result={label(results[0]) if results else 'None'}"
            )

    count = len(EVALUATION_CASES)

    print("\n" + "=" * 55)
    print("FINAL AVERAGE METRICS")
    print("=" * 55)

    for method, values in totals.items():
        print(
            f"{method:7} | "
            f"Mean Recall@{TOP_K}="
            f"{values['recall'] / count:.3f} | "
            f"Mean MRR@{TOP_K}="
            f"{values['mrr'] / count:.3f}"
        )


if __name__ == "__main__":
    main()
