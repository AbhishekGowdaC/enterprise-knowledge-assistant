
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "enterprise_documents"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(url="http://localhost:6333")

    query = input("Enter your question: ").strip()

    if not query:
        print("Please enter a question.")
        return

    # Convert the question into the same vector space as the documents.
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    # Retrieve the most semantically similar document chunks.
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,
        with_payload=True,
    ).points

    if not results:
        print("No matching chunks found.")
        return

    for rank, point in enumerate(results, start=1):
        payload = point.payload or {}

        print(f"\n--- Result {rank} ---")
        print(f"Similarity score: {point.score:.4f}")
        print(f"Source: {payload.get('source', 'unknown')}")
        print(f"Chunk index: {payload.get('chunk_index', 'unknown')}")
        print(f"Text: {payload.get('text', '')}")


if __name__ == "__main__":
    main()
