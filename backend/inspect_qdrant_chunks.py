
from qdrant_client import QdrantClient

COLLECTION_NAME = "enterprise_documents"

client = QdrantClient(url="http://localhost:6333")

try:
    offset = None
    total = 0

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            source = payload.get("source", "unknown")
            chunk_index = payload.get("chunk_index", "unknown")
            text = payload.get("text", "").strip()

            print("=" * 80)
            print(f"Point ID: {point.id}")
            print(f"Source: {source}")
            print(f"Chunk index: {chunk_index}")
            print(f"Text: {text[:700]}")

            total += 1

        if offset is None:
            break

    print("\nTotal chunks inspected:", total)

finally:
    client.close()
