import csv
import json
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import app.retrieval.hybrid_search as hs


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

DATASET_PATH = (
    PROJECT_ROOT / "data/evaluation/final/expanded_golden_dataset_v1.json"
)
OUTPUT_DIR = PROJECT_ROOT / "data/evaluation/final"

COLLECTION_NAME = "enterprise_documents_expanded"
K_VALUES = (1, 3, 5)


def load_dataset():
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    questions = [
        item for item in data["questions"]
        if item["answerable"] is True
    ]

    return questions


def label(result):
    return result["source"], int(result["chunk_index"])


def precision_at_k(results, relevant, k):
    return len({label(r) for r in results[:k]} & relevant) / k


def recall_at_k(results, relevant, k):
    return len({label(r) for r in results[:k]} & relevant) / len(relevant)


def mrr_at_k(results, relevant, k):
    for rank, result in enumerate(results[:k], start=1):
        if label(result) in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_method(method_name, questions, retrieve):
    rows = []

    for number, item in enumerate(questions, start=1):
        results = retrieve(item["question"])

        relevant = {
            (chunk["source"], int(chunk["chunk_index"]))
            for chunk in item["relevant_chunks"]
        }

        row = {
            "question_id": item["id"],
            "question": item["question"],
        }

        row["retrieved_top_5"] = json.dumps([
            {
                "source": r["source"],
                "chunk_index": int(r["chunk_index"]),
                "score": round(
                    float(r.get(
                        "dense_score",
                        r.get("bm25_score", r.get("rrf_score", 0.0))
                    )),
                    6,
                ),
            }
            for r in results[:5]
        ], ensure_ascii=False)

        for k in K_VALUES:
            row[f"precision_at_{k}"] = precision_at_k(
                results, relevant, k
            )
            row[f"recall_at_{k}"] = recall_at_k(
                results, relevant, k
            )
            row[f"mrr_at_{k}"] = mrr_at_k(
                results, relevant, k
            )

        rows.append(row)

        print(
            f"{method_name:6} [{number:02d}/{len(questions)}] "
            f"P@5={row['precision_at_5']:.3f} "
            f"R@5={row['recall_at_5']:.3f} "
            f"MRR@5={row['mrr_at_5']:.3f} "
            f"| {item['id']}"
        )

    summary = {
        "method": method_name,
        "questions_evaluated": len(rows),
    }

    for k in K_VALUES:
        for metric in ("precision", "recall", "mrr"):
            key = f"{metric}_at_{k}"
            summary[f"mean_{key}"] = mean(
                row[key] for row in rows
            )

    return rows, summary


def save_results(method_name, rows, summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_question_path = (
        OUTPUT_DIR / f"expanded_{method_name}_per_question.csv"
    )
    summary_path = (
        OUTPUT_DIR / f"expanded_{method_name}_summary.csv"
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


def main():
    # Change only the imported module's collection setting in this process.
    # The production application and source file remain unchanged.
    hs.COLLECTION_NAME = COLLECTION_NAME

    questions = load_dataset()

    client = QdrantClient(url=hs.QDRANT_URL)
    collection_info = client.get_collection(COLLECTION_NAME)

    print("=== EXPANDED RETRIEVAL COMPARISON ===")
    print("Questions:", len(questions))
    print("Collection:", COLLECTION_NAME)
    print("Indexed chunks:", collection_info.points_count)

    print("\nLoading embedding model...")
    model = SentenceTransformer(hs.EMBEDDING_MODEL)

    print("Loading indexed chunks...")
    chunks = hs.load_chunks(client)
    print("Loaded chunks:", len(chunks))

    if len(chunks) != collection_info.points_count:
        raise ValueError("Loaded chunk count does not match Qdrant.")

    bm25 = BM25Okapi([
        hs.tokenize(chunk["text"])
        for chunk in chunks
    ])

    def dense_retrieve(question):
        return hs.dense_search(question, model, client)

    def bm25_retrieve(question):
        return hs.bm25_search(question, chunks, bm25)

    def hybrid_retrieve(question):
        dense_results = hs.dense_search(question, model, client)
        bm25_results = hs.bm25_search(question, chunks, bm25)
        exact_candidates = hs.exact_match_search(
            question, chunks, bm25
        )

        return hs.reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            question,
            exact_candidates,
        )

    all_summaries = []

    for method_name, retrieve in (
        ("dense", dense_retrieve),
        ("bm25", bm25_retrieve),
        ("hybrid", hybrid_retrieve),
    ):
        print(f"\n=== Evaluating {method_name.upper()} ===")

        rows, summary = evaluate_method(
            method_name, questions, retrieve
        )

        save_results(method_name, rows, summary)
        all_summaries.append(summary)

    comparison_path = OUTPUT_DIR / "expanded_retrieval_comparison.csv"

    with comparison_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_summaries)

    print("\n=== FINAL COMPARISON ===")

    for summary in all_summaries:
        print(f"\n{summary['method'].upper()}")

        for k in K_VALUES:
            print(
                f"K={k} | "
                f"P={summary[f'mean_precision_at_{k}']:.4f} | "
                f"R={summary[f'mean_recall_at_{k}']:.4f} | "
                f"MRR={summary[f'mean_mrr_at_{k}']:.4f}"
            )

    print("\nComparison saved:", comparison_path)


if __name__ == "__main__":
    main()
