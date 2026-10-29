import csv
import json
import os
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# Resolve the project root and load configuration.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

DATASET_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_dataset.json"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"

COLLECTION_NAME = "enterprise_documents"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

K_VALUES = (1, 3, 5)


def chunk_label(payload):
    """Return a stable identifier for a document chunk."""
    source = payload.get("source")
    index = payload.get("chunk_index")

    if source is None or index is None:
        return None

    try:
        index = int(index)
    except (TypeError, ValueError):
        return None

    return source, index


def load_dataset():
    """Load answerable questions and their labeled relevant chunks."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Retrieval dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    questions = data.get("questions", [])

    if not questions:
        raise ValueError("The retrieval dataset contains no questions.")

    for item in questions:
        if not item.get("relevant_chunks"):
            raise ValueError(
                f"No relevant chunks labeled for: {item.get('question')}"
            )

    return questions


def get_indexed_labels(client):
    """Read indexed source/chunk identifiers for label verification."""
    indexed = set()
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=["source", "chunk_index"],
            with_vectors=False,
        )

        for point in points:
            label = chunk_label(point.payload or {})
            if label is not None:
                indexed.add(label)

        if next_offset is None:
            break

        offset = next_offset

    return indexed


def precision_at_k(retrieved, relevant, k):
    """Fraction of the top-k positions occupied by relevant chunks."""
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / k


def recall_at_k(retrieved, relevant, k):
    """Fraction of labeled relevant chunks found in the top k."""
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(relevant)


def reciprocal_rank_at_k(retrieved, relevant, k):
    """Reciprocal rank of the first relevant result within the top k."""
    for rank, label in enumerate(retrieved[:k], start=1):
        if label in relevant:
            return 1.0 / rank

    return 0.0


def main():
    questions = load_dataset()

    print(f"Dataset: {DATASET_PATH}")
    print(f"Questions: {len(questions)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"K values: {K_VALUES}")

    client = QdrantClient(url=QDRANT_URL)

    # Verify all manual labels refer to real indexed chunks.
    indexed_labels = get_indexed_labels(client)
    missing_labels = []

    for item in questions:
        for relevant_chunk in item["relevant_chunks"]:
            label = (
                relevant_chunk["source"],
                int(relevant_chunk["chunk_index"]),
            )

            if label not in indexed_labels:
                missing_labels.append((item["question"], label))

    if missing_labels:
        print("\nERROR: Some labeled chunks were not found in Qdrant.")

        for question, label in missing_labels:
            print(f"  Question: {question}")
            print(f"  Missing label: {label}")

        raise SystemExit(
            "Verify the source filenames and chunk indices before evaluation."
        )

    print(f"Indexed chunks verified: {len(indexed_labels)}")
    print("Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    per_question_rows = []

    for number, item in enumerate(questions, start=1):
        question = item["question"]

        relevant = {
            (
                chunk["source"],
                int(chunk["chunk_index"]),
            )
            for chunk in item["relevant_chunks"]
        }

        # Use the same embedding configuration as the application.
        query_vector = model.encode(
            question,
            normalize_embeddings=True,
        ).tolist()

        # Retrieve the raw top 5. Do not apply the answer abstention
        # threshold here: this evaluates ranking quality separately.
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=max(K_VALUES),
            with_payload=True,
        )

        points = response.points

        retrieved = []
        retrieved_details = []

        for point in points:
            payload = point.payload or {}
            label = chunk_label(payload)

            if label is None:
                continue

            retrieved.append(label)
            retrieved_details.append({
                "source": label[0],
                "chunk_index": label[1],
                "score": round(float(point.score), 6),
            })

        row = {
            "question": question,
            "expected_source": item.get("expected_source", ""),
            "relevant_chunk_count": len(relevant),
            "relevant_chunks": json.dumps(
                sorted(
                    [
                        {"source": source, "chunk_index": index}
                        for source, index in relevant
                    ],
                    key=lambda chunk: (
                        chunk["source"],
                        chunk["chunk_index"],
                    ),
                ),
                ensure_ascii=False,
            ),
            "retrieved_top_5": json.dumps(
                retrieved_details,
                ensure_ascii=False,
            ),
        }

        for k in K_VALUES:
            row[f"precision_at_{k}"] = precision_at_k(
                retrieved, relevant, k
            )
            row[f"recall_at_{k}"] = recall_at_k(
                retrieved, relevant, k
            )
            row[f"mrr_at_{k}"] = reciprocal_rank_at_k(
                retrieved, relevant, k
            )

        per_question_rows.append(row)

        p5 = row["precision_at_5"]
        r5 = row["recall_at_5"]
        m5 = row["mrr_at_5"]

        print(
            f"[{number:02d}/{len(questions)}] "
            f"P@5={p5:.3f}  R@5={r5:.3f}  "
            f"RR@5={m5:.3f}  | {question}"
        )

    metric_names = [
        f"{metric}_at_{k}"
        for k in K_VALUES
        for metric in ("precision", "recall", "mrr")
    ]

    summary = {
        "questions_evaluated": len(per_question_rows),
    }

    for metric_name in metric_names:
        summary[f"mean_{metric_name}"] = mean(
            row[metric_name] for row in per_question_rows
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_question_path = (
        OUTPUT_DIR / "retrieval_metrics_per_question.csv"
    )
    summary_path = OUTPUT_DIR / "retrieval_metrics_summary.csv"

    with per_question_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(per_question_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(per_question_rows)

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\n=== Mean Retrieval Metrics ===")
    print(f"Questions evaluated: {len(per_question_rows)}")

    for k in K_VALUES:
        print(
            f"\nK = {k}"
            f"\n  Precision@{k}: "
            f"{summary[f'mean_precision_at_{k}']:.4f}"
            f"\n  Recall@{k}: "
            f"{summary[f'mean_recall_at_{k}']:.4f}"
            f"\n  MRR@{k}: "
            f"{summary[f'mean_mrr_at_{k}']:.4f}"
        )

    print("\nSaved per-question results:", per_question_path)
    print("Saved summary results:", summary_path)


if __name__ == "__main__":
    main()
