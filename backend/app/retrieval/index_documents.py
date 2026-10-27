
from pathlib import Path
import uuid

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from app.ingestion.parsers import parse_html
from app.ingestion.chunking import chunk_records


BASE_DIR = Path(__file__).resolve().parents[3]
SAMPLE_FILE = BASE_DIR / "data" / "raw" / "sample.html"

COLLECTION_NAME = "enterprise_documents"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def main():
    # 1. Parse the source document.
    records = parse_html(str(SAMPLE_FILE))
    print(f"Parsed records: {len(records)}")

    # 2. Split text into overlapping chunks.
    chunks = chunk_records(
        records,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No text chunks found.")

    # 3. Load the embedding model.
    # First run downloads the model if it is not cached.
    model = SentenceTransformer(MODEL_NAME)

    # 4. Convert each chunk into a normalized vector.
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"Embedding shape: {embeddings.shape}")

    # 5. Connect to Qdrant.
    client = QdrantClient(url="http://localhost:6333")

    # 6. Create the collection if it doesn't exist.
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embeddings.shape[1],
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Created collection: {COLLECTION_NAME}")

    # 7. Store vectors and their metadata.
    points = []

    for chunk, embedding in zip(chunks, embeddings):
        # Deterministic ID helps avoid duplicate points on reruns.
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f'{chunk["source"]}:{chunk["chunk_index"]}',
            )
        )

        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=chunk,
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    print(f"Indexed {len(points)} chunks successfully.")

    # 8. Verify the stored point count.
    info = client.get_collection(COLLECTION_NAME)
    print(f"Points in collection: {info.points_count}")


if __name__ == "__main__":
    main()
