
import csv
import json
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
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
CANDIDATE_K = 30
BATCH_SIZE = 16


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


def evaluate_results(results, relevant):
    metrics = {}

    for k in K_VALUES:
        metrics[f"precision_at_{k}"] = precision_at_k(
            results, relevant, k
        )
        metrics[f"recall_at_{k}"] = recall_at_k(
            results, relevant, k
        )
        metrics[f"mrr_at_{k}"] = mrr_at_k(
            results, relevant, k
        )

    return metrics


def main():
    hs.COLLECTION_NAME = COLLECTION_NAME

    questions = load_dataset()

    client = QdrantClient(url=hs.QDRANT_URL)
    collection_info = client.get_collection(COLLECTION_NAME)

    print("=== CROSS-ENCODER BENCHMARK ===")
    print("Total answerable questions:", len(questions))
    print("Collection:", COLLECTION_NAME)
    print("Indexed chunks:", collection_info.points_count)

    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer(hs.EMBEDDING_MODEL)

    print("Loading indexed chunks...")
    chunks = hs.load_chunks(client)
    print("Loaded chunks:", len(chunks))

    if len(chunks) != collection_info.points_count:
        raise ValueError("Loaded chunk count does not match Qdrant.")

    bm25 = BM25Okapi([
        hs.tokenize(chunk["text"])
        for chunk in chunks
    ])

    print("\nLoading Cross-Encoder on CPU...")
    cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
    )

    baseline_rows = []
    reranked_rows = []

    for number, item in enumerate(questions, start=1):
        question = item["question"]

        relevant = {
            (chunk["source"], int(chunk["chunk_index"]))
            for chunk in item["relevant_chunks"]
        }

        # Stage 1: Retrieve candidates using existing Hybrid RRF.
        dense_results = hs.dense_search(
            question, embedding_model, client
        )
        bm25_results = hs.bm25_search(
            question, chunks, bm25
        )
        exact_candidates = hs.exact_match_search(
            question, chunks, bm25
        )

        fused_results = hs.reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            question,
            exact_candidates,
        )

        candidates = fused_results[:CANDIDATE_K]

        # Stage 2: Measure candidate recall before reranking.
        candidate_recall = (
            len({label(r) for r in candidates} & relevant)
            / len(relevant)
        )

        # Stage 3: Cross-Encoder scores query-document pairs.
        pairs = [
            (question, result["text"])
            for result in candidates
        ]

        scores = cross_encoder.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )

        # Stage 4: Sort candidates by Cross-Encoder score.
        scored_results = []

        for result, score in zip(candidates, scores):
            updated = dict(result)
            updated["cross_encoder_score"] = float(score)
            scored_results.append(updated)

        reranked_results = sorted(
            scored_results,
            key=lambda r: r["cross_encoder_score"],
            reverse=True,
        )

        # Evaluate both methods using the same candidate pool.
        baseline_metrics = evaluate_results(
            candidates, relevant
        )
        reranked_metrics = evaluate_results(
            reranked_results, relevant
        )

        common = {
            "question_id": item["id"],
            "question": question,
            "candidate_recall_at_30": candidate_recall,
        }

        baseline_row = {
            **common,
            **baseline_metrics,
            "retrieved_top_5": json.dumps([
                {
                    "source": r["source"],
                    "chunk_index": int(r["chunk_index"]),
                    "rrf_score": round(
                        float(r.get("rrf_score", 0.0)), 6
                    ),
                }
                for r in candidates[:5]
            ], ensure_ascii=False),
        }

        reranked_row = {
            **common,
            **reranked_metrics,
            "retrieved_top_5": json.dumps([
                {
                    "source": r["source"],
                    "chunk_index": int(r["chunk_index"]),
                    "cross_encoder_score": round(
                        r["cross_encoder_score"], 6
                    ),
                    "rrf_score": round(
                        float(r.get("rrf_score", 0.0)), 6
                    ),
                }
                for r in reranked_results[:5]
            ], ensure_ascii=False),
        }

        baseline_rows.append(baseline_row)
        reranked_rows.append(reranked_row)

        print(
            f"[{number:02d}/{len(questions)}] {item['id']} "
            f"CandidateRecall@30={candidate_recall:.3f} "
            f"Hybrid MRR@5={baseline_metrics['mrr_at_5']:.3f} "
            f"Reranked MRR@5={reranked_metrics['mrr_at_5']:.3f}"
        )

    # Save per-question results.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, rows in (
        ("hybrid_candidate", baseline_rows),
        ("cross_encoder", reranked_rows),
    ):
        path = OUTPUT_DIR / f"expanded_{name}_per_question.csv"

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    # Calculate summary metrics.
    summaries = []

    for name, rows in (
        ("hybrid_candidate", baseline_rows),
        ("cross_encoder", reranked_rows),
    ):
        summary = {
            "method": name,
            "questions_evaluated": len(rows),
            "mean_candidate_recall_at_30": mean(
                row["candidate_recall_at_30"] for row in rows
            ),
        }

        for k in K_VALUES:
            for metric in ("precision", "recall", "mrr"):
                key = f"{metric}_at_{k}"
                summary[f"mean_{key}"] = mean(
                    row[key] for row in rows
                )

        summaries.append(summary)

    summary_path = OUTPUT_DIR / "expanded_cross_encoder_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summaries)

    # Print final comparison.
    print("\n=== FINAL HYBRID vs CROSS-ENCODER ===")

    for summary in summaries:
        print(f"\n{summary['method'].upper()}")
        print(
            "Candidate Recall@30:",
            f"{summary['mean_candidate_recall_at_30']:.4f}",
        )

        for k in K_VALUES:
            print(
                f"K={k} | "
                f"P={summary[f'mean_precision_at_{k}']:.4f} | "
                f"R={summary[f'mean_recall_at_{k}']:.4f} | "
                f"MRR={summary[f'mean_mrr_at_{k}']:.4f}"
            )

    print("\nSaved benchmark results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
