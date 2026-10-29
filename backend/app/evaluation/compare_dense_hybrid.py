import csv
import json
from pathlib import Path
from statistics import mean

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from app.retrieval.hybrid_search import (
    load_chunks,
    dense_search,
    bm25_search,
    exact_match_search,
    reciprocal_rank_fusion,
    tokenize,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_dataset.json"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "enterprise_documents"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
K_VALUES = (1, 3, 5)


def get_label(result):
    """Create a stable (source, chunk_index) identifier."""
    return (
        result["source"],
        int(result["chunk_index"]),
    )


def precision_at_k(results, relevant, k):
    retrieved = {get_label(item) for item in results[:k]}
    return len(retrieved & relevant) / k


def recall_at_k(results, relevant, k):
    retrieved = {get_label(item) for item in results[:k]}
    return len(retrieved & relevant) / len(relevant)


def mrr_at_k(results, relevant, k):
    for rank, item in enumerate(results[:k], start=1):
        if get_label(item) in relevant:
            return 1.0 / rank
    return 0.0


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Retrieval dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    questions = dataset["questions"]

    if not questions:
        raise ValueError("The retrieval dataset contains no questions.")

    print(f"Dataset: {DATASET_PATH}")
    print(f"Questions: {len(questions)}")

    client = QdrantClient(url=QDRANT_URL)

    print("Loading indexed chunks...")
    chunks = load_chunks(client)

    if not chunks:
        raise RuntimeError("No chunks found in Qdrant.")

    print(f"Indexed chunks loaded: {len(chunks)}")

    # Validate the ground-truth labels against the current index.
    indexed_labels = {get_label(chunk) for chunk in chunks}
    missing_labels = []

    for item in questions:
        for chunk in item["relevant_chunks"]:
            label = (
                chunk["source"],
                int(chunk["chunk_index"]),
            )
            if label not in indexed_labels:
                missing_labels.append((item["question"], label))

    if missing_labels:
        print("\nERROR: Ground-truth labels missing from Qdrant:")
        for question, label in missing_labels:
            print(f"  {question}: {label}")
        raise SystemExit("Correct the labels before comparing methods.")

    print("All ground-truth labels verified.")
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])

    methods = ("Dense", "BM25", "Hybrid")
    rows = []

    for number, item in enumerate(questions, start=1):
        question = item["question"]

        relevant = {
            (
                chunk["source"],
                int(chunk["chunk_index"]),
            )
            for chunk in item["relevant_chunks"]
        }

        # All three methods receive exactly the same question.
        dense_results = dense_search(question, model, client)
        keyword_results = bm25_search(question, chunks, bm25)
        exact_results = exact_match_search(question, chunks, bm25)

        hybrid_results = reciprocal_rank_fusion(
            dense_results,
            keyword_results,
            question,
            exact_candidates=exact_results,
        )

        results_by_method = {
            "Dense": dense_results,
            "BM25": keyword_results,
            "Hybrid": hybrid_results,
        }

        row = {
            "question": question,
            "expected_source": item.get("expected_source", ""),
            "relevant_chunks": json.dumps(
                [
                    {"source": source, "chunk_index": index}
                    for source, index in sorted(relevant)
                ],
                ensure_ascii=False,
            ),
        }

        display_metrics = []

        for method in methods:
            results = results_by_method[method]

            # Record top-five results to help diagnose ranking failures.
            row[f"{method.lower()}_top_5"] = json.dumps(
                [
                    {
                        "source": result["source"],
                        "chunk_index": int(result["chunk_index"]),
                    }
                    for result in results[:5]
                ],
                ensure_ascii=False,
            )

            for k in K_VALUES:
                row[f"{method.lower()}_precision_at_{k}"] = (
                    precision_at_k(results, relevant, k)
                )
                row[f"{method.lower()}_recall_at_{k}"] = (
                    recall_at_k(results, relevant, k)
                )
                row[f"{method.lower()}_mrr_at_{k}"] = (
                    mrr_at_k(results, relevant, k)
                )

            display_metrics.append(
                f"{method} R@5="
                f"{row[f'{method.lower()}_recall_at_5']:.2f}, "
                f"MRR@5="
                f"{row[f'{method.lower()}_mrr_at_5']:.2f}"
            )

        rows.append(row)

        print(f"\n[{number:02d}/{len(questions)}] {question}")
        print(" | ".join(display_metrics))

    # Calculate the macro-average over the same 30 questions.
    summary = {"questions_evaluated": len(rows)}

    for method in methods:
        prefix = method.lower()

        for k in K_VALUES:
            for metric in ("precision", "recall", "mrr"):
                column = f"{prefix}_{metric}_at_{k}"
                summary[f"mean_{column}"] = mean(
                    row[column] for row in rows
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    details_path = OUTPUT_DIR / "dense_hybrid_comparison_details.csv"
    summary_path = OUTPUT_DIR / "dense_hybrid_comparison_summary.csv"

    with details_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary.keys()),
        )
        writer.writeheader()
        writer.writerow(summary)

    print("\n" + "=" * 72)
    print("MEAN RETRIEVAL METRICS — SAME QUESTIONS AND LABELS")
    print("=" * 72)

    for method in methods:
        prefix = method.lower()
        print(f"\n{method}")

        for k in K_VALUES:
            precision = summary[f"mean_{prefix}_precision_at_{k}"]
            recall = summary[f"mean_{prefix}_recall_at_{k}"]
            mrr = summary[f"mean_{prefix}_mrr_at_{k}"]

            print(
                f"  K={k}: "
                f"Precision={precision:.4f}, "
                f"Recall={recall:.4f}, "
                f"MRR={mrr:.4f}"
            )

    print("\nSaved detailed results:", details_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
