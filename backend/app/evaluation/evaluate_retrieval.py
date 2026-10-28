
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_PATH = BASE_DIR / "data" / "evaluation" / "golden_dataset.json"

COLLECTION_NAME = "enterprise_documents"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Keep these aligned with the application's retrieval settings.
TOP_K = 5
MIN_SIMILARITY = 0.45


def main():
    with DATASET_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(url="http://localhost:6333")

    answerable_count = 0
    source_hits = 0
    reciprocal_rank_sum = 0.0

    unanswerable_count = 0
    threshold_rejections = 0

    try:
        for item in dataset:
            question = item["question"]
            expected_source = item.get("expected_source", "")
            answerable = item["answerable"]

            query_vector = model.encode(
                question,
                normalize_embeddings=True,
            ).tolist()

            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=TOP_K,
                with_payload=True,
            ).points

            retrieved_sources = []

            for rank, point in enumerate(results, start=1):
                payload = point.payload or {}
                source = payload.get("source", "unknown")
                chunk_index = payload.get("chunk_index", "unknown")
                text = payload.get("text", "").strip()

                retrieved_sources.append(source)

                print(f"\n  Rank: {rank}")
                print(f"  Source: {source}")
                print(f"  Chunk: {chunk_index}")
                print(f"  Score: {point.score:.4f}")
                print(f"  Text preview: {text[:250]!r}")

            top_score = results[0].score if results else None

            # This matches the application's similarity-threshold gate.
            rejected_by_threshold = (
                top_score is None
                or top_score < MIN_SIMILARITY
            )

            source_found = (
                answerable
                and bool(expected_source)
                and expected_source in retrieved_sources
            )

            if answerable:
                answerable_count += 1
                source_hits += int(source_found)

                if source_found:
                    rank = retrieved_sources.index(expected_source) + 1
                    reciprocal_rank_sum += 1 / rank
                else:
                    rank = None
            else:
                unanswerable_count += 1
                threshold_rejections += int(rejected_by_threshold)
                rank = None

            print("\n" + "=" * 65)
            print(f"Question: {question}")
            print(f"Answerable in dataset: {answerable}")
            print(f"Expected source: {expected_source or 'None'}")
            print(f"Retrieved sources: {retrieved_sources}")
            print(
                f"Top similarity score: "
                f"{top_score:.4f}" if top_score is not None
                else "Top similarity score: No results"
            )
            print(f"Expected source found in top {TOP_K}: {source_found}")

            if answerable:
                print(f"Expected source rank: {rank or 'Not found'}")
            else:
                print(
                    "Threshold would reject retrieval: "
                    f"{rejected_by_threshold}"
                )

    finally:
        client.close()

    hit_rate = (
        source_hits / answerable_count
        if answerable_count
        else 0.0
    )

    mrr = (
        reciprocal_rank_sum / answerable_count
        if answerable_count
        else 0.0
    )

    print("\n" + "=" * 65)
    print("RETRIEVAL EVALUATION SUMMARY")
    print(f"Total questions: {len(dataset)}")
    print(f"Answerable questions: {answerable_count}")
    print(f"Unanswerable questions: {unanswerable_count}")
    print(f"Source Hit Rate@{TOP_K}: {hit_rate:.2%}")
    print(f"Source MRR@{TOP_K}: {mrr:.3f}")

    if unanswerable_count:
        print(
            "Unanswerable questions rejected by threshold: "
            f"{threshold_rejections}/{unanswerable_count} "
            f"({threshold_rejections / unanswerable_count:.2%})"
        )
        print(
            "Note: threshold rejection is only a retrieval diagnostic; "
            "it does not prove that the retrieved context can answer "
            "a question."
        )


if __name__ == "__main__":
    main()
