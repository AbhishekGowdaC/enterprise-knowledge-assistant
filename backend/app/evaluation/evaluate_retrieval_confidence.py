import csv
import json
from pathlib import Path
from statistics import mean, median

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
    PROJECT_ROOT / "data/evaluation/golden_dataset.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data/evaluation/retrieval_confidence_diagnostics.csv"
)

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "enterprise_documents"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_dataset():
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    questions = data if isinstance(data, list) else data["questions"]

    if not questions:
        raise ValueError("Golden dataset is empty.")

    return questions


def get_score_statistics(values):
    if not values:
        return None

    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main():
    questions = load_dataset()

    print("Questions:", len(questions))
    print("Loading Qdrant chunks...")

    client = QdrantClient(url=QDRANT_URL)

    try:
        chunks = load_chunks(client)

        if not chunks:
            raise RuntimeError("No chunks found in Qdrant.")

        print("Indexed chunks:", len(chunks))

        print("Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME)

        bm25 = BM25Okapi([
            tokenize(chunk["text"])
            for chunk in chunks
        ])

        rows = []

        for index, item in enumerate(questions, start=1):
            question = item["question"]
            answerable = bool(item["answerable"])

            dense_results = dense_search(
                question, model, client
            )

            bm25_results = bm25_search(
                question, chunks, bm25
            )

            exact_results = exact_match_search(
                question, chunks, bm25
            )

            hybrid_results = reciprocal_rank_fusion(
                dense_results,
                bm25_results,
                question,
                exact_candidates=exact_results,
            )

            dense_top = dense_results[0] if dense_results else None
            hybrid_top = hybrid_results[0] if hybrid_results else None

            dense_score = (
                dense_top["dense_score"] if dense_top else ""
            )

            hybrid_score = (
                hybrid_top["rrf_score"] if hybrid_top else ""
            )

            dense_rank = (
                hybrid_top.get("dense_rank") if hybrid_top else None
            )

            bm25_rank = (
                hybrid_top.get("bm25_rank") if hybrid_top else None
            )

            appears_in_both = (
                dense_rank is not None and bm25_rank is not None
                if hybrid_top else False
            )

            row = {
                "question": question,
                "answerable": answerable,
                "dense_top1_score": dense_score,
                "dense_top1_source": (
                    dense_top["source"] if dense_top else ""
                ),
                "dense_top1_chunk": (
                    dense_top["chunk_index"] if dense_top else ""
                ),
                "hybrid_top1_rrf_score": hybrid_score,
                "hybrid_top1_source": (
                    hybrid_top["source"] if hybrid_top else ""
                ),
                "hybrid_top1_chunk": (
                    hybrid_top["chunk_index"] if hybrid_top else ""
                ),
                "hybrid_top1_dense_rank": dense_rank or "",
                "hybrid_top1_bm25_rank": bm25_rank or "",
                "hybrid_top1_in_both": appears_in_both,
            }

            rows.append(row)

            print(
                f"[{index}/{len(questions)}] "
                f"{'ANSWERABLE' if answerable else 'UNANSWERABLE'} | "
                f"Dense={dense_score if dense_score != '' else 'None'} | "
                f"Hybrid RRF={hybrid_score if hybrid_score != '' else 'None'}"
            )

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with OUTPUT_PATH.open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(rows)

        print("\n=== CONFIDENCE SCORE DISTRIBUTION ===")

        for group_name, group in (
            ("ANSWERABLE", [r for r in rows if r["answerable"]]),
            ("UNANSWERABLE", [r for r in rows if not r["answerable"]]),
        ):
            print(f"\n{group_name}")

            for field in (
                "dense_top1_score",
                "hybrid_top1_rrf_score",
            ):
                scores = [
                    float(r[field])
                    for r in group
                    if r[field] != ""
                ]

                stats = get_score_statistics(scores)

                print(f"{field}: {stats}")

        print("\n=== HYBRID RANKING AGREEMENT ===")

        for group_name, group in (
            ("ANSWERABLE", [r for r in rows if r["answerable"]]),
            ("UNANSWERABLE", [r for r in rows if not r["answerable"]]),
        ):
            total = len(group)
            agreement = sum(
                r["hybrid_top1_in_both"] for r in group
            )

            print(
                f"{group_name}: {agreement}/{total} "
                "top hybrid results appeared in both retrievers"
            )

        print("\nSaved diagnostic results:", OUTPUT_PATH)

    finally:
        client.close()


if __name__ == "__main__":
    main()
