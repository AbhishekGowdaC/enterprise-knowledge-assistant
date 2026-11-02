
import re
from pathlib import Path

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "enterprise_documents_expanded"
TOP_K = 5


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks(client: QdrantClient) -> list[dict]:
    """Load indexed text chunks from Qdrant."""
    chunks = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            text = payload.get("text", "").strip()

            if text:
                chunks.append(
                    {
                        "text": text,
                        "source": payload.get("source", "unknown"),
                        "chunk_index": payload.get("chunk_index", -1),
                    }
                )

        if next_offset is None:
            break

        offset = next_offset

    return chunks


def main():
    client = QdrantClient(url=QDRANT_URL)

    try:
        chunks = load_chunks(client)

        if not chunks:
            raise RuntimeError(
                "No text chunks found. Check your Qdrant index."
            )

        tokenized_corpus = [
            tokenize(chunk["text"]) for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_corpus)

        print(f"Loaded chunks: {len(chunks)}")
        print("BM25 keyword retrieval is ready.")

        while True:
            question = input(
                "\nEnter a question (or type 'exit'): "
            ).strip()

            if question.lower() == "exit":
                break

            query_tokens = tokenize(question)

            if not query_tokens:
                print("Please enter a question containing words.")
                continue

            scores = bm25.get_scores(query_tokens)

            # Sort by score, highest first.
            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:TOP_K]

            print("\nTop BM25 results:")

            for rank, index in enumerate(ranked_indices, start=1):
                chunk = chunks[index]

                print(f"\nRank: {rank}")
                print(f"BM25 score: {scores[index]:.4f}")
                print(f"Source: {chunk['source']}")
                print(f"Chunk: {chunk['chunk_index']}")
                print(f"Text: {chunk['text'][:350]!r}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
