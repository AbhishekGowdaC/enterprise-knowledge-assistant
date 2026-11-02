import csv
import json
import os
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "final"
    / "expanded_golden_dataset_v1.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "final"

COLLECTION_NAME = "enterprise_documents_expanded"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

K_VALUES = (1, 3, 5)


def chunk_label(payload):
    source = payload.get("source")
    index = payload.get("chunk_index")

    if source is None or index is None:
        return None

    try:
        return source, int(index)
    except (TypeError, ValueError):
        return None


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    questions = data.get("questions", [])

    answerable = [
        item for item in questions
        if item.get("answerable") is True
    ]

    if not answerable:
        raise ValueError("No answerable questions found.")

    return answerable


def precision_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(retrieved, relevant, k):
    for rank, label in enumerate(retrieved[:k], start=1):
        if label in relevant:
            return 1.0 / rank
    return 0.0


def main():
    questions = load_dataset()

    print("=== Expanded Dense Retrieval Evaluation ===")
    print("Dataset:", DATASET_PATH)
    print("Collection:", COLLECTION_NAME)
    print("Answerable questions:", len(questions))
    print("K values:", K_VALUES)

    client = QdrantClient(url=QDRANT_URL)

    collection_info = client.get_collection(COLLECTION_NAME)
    print("Indexed points:", collection_info.points_count)

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

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

        query_vector = model.encode(
            question,
            normalize_embeddings=True,
        ).tolist()

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=max(K_VALUES),
            with_payload=True,
        )

        retrieved = []
        details = []

        for point in response.points:
            label = chunk_label(point.payload or {})

            if label is None:
                continue

            retrieved.append(label)

            details.append({
                "source": label[0],
                "chunk_index": label[1],
                "score": round(float(point.score), 6),
            })

        row = {
            "question_id": item["id"],
            "question": question,
            "question_type": item.get("question_type", ""),
            "relevant_chunk_count": len(relevant),
            "relevant_chunks": json.dumps([
                {
                    "source": source,
                    "chunk_index": index,
                }
                for source, index in sorted(relevant)
            ], ensure_ascii=False),
            "retrieved_top_5": json.dumps(
                details,
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

        rows.append(row)

        print(
            f"[{number:02d}/{len(questions)}] "
            f"P@5={row['precision_at_5']:.3f} "
            f"R@5={row['recall_at_5']:.3f} "
            f"MRR@5={row['mrr_at_5']:.3f} "
            f"| {item['id']}"
        )

    metric_names = [
        f"{metric}_at_{k}"
        for k in K_VALUES
        for metric in ("precision", "recall", "mrr")
    ]

    summary = {
        "dataset": DATASET_PATH.name,
        "collection": COLLECTION_NAME,
        "retrieval_method": "dense",
        "questions_evaluated": len(rows),
    }

    for metric_name in metric_names:
        summary[f"mean_{metric_name}"] = mean(
            row[metric_name] for row in rows
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_question_path = (
        OUTPUT_DIR / "expanded_dense_per_question.csv"
    )
    summary_path = (
        OUTPUT_DIR / "expanded_dense_summary.csv"
    )

    with per_question_path.open(
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

    print("\n=== DENSE RETRIEVAL SUMMARY ===")
    print("Questions evaluated:", len(rows))

    for k in K_VALUES:
        print(
            f"\nK = {k}"
            f"\nPrecision@{k}: "
            f"{summary[f'mean_precision_at_{k}']:.4f}"
            f"\nRecall@{k}: "
            f"{summary[f'mean_recall_at_{k}']:.4f}"
            f"\nMRR@{k}: "
            f"{summary[f'mean_mrr_at_{k}']:.4f}"
        )

    print("\nSaved:")
    print(per_question_path)
    print(summary_path)


if __name__ == "__main__":
    main()
