import argparse
import uuid
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from app.ingestion.parsers import parse_pdf, parse_html
from app.ingestion.chunking import chunk_records


PROJECT_ROOT = Path(__file__).resolve().parents[3]

ORIGINAL_DIR = PROJECT_ROOT / "data" / "raw"
EXPANDED_DIR = PROJECT_ROOT / "data" / "expanded_raw"

COLLECTION_NAME = "enterprise_documents_expanded"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
BATCH_SIZE = 32


def load_documents():
    """Load original PDFs and additional synthetic HTML documents."""
    records = []

    for directory in (ORIGINAL_DIR, EXPANDED_DIR):
        if not directory.exists():
            raise FileNotFoundError(directory)

        for path in sorted(directory.glob("*.pdf")):
            print(f"Parsing PDF: {path.name}")
            records.extend(parse_pdf(str(path)))

        for path in sorted(directory.glob("*.html")):
            print(f"Parsing HTML: {path.name}")
            records.extend(parse_html(str(path)))

    return records


def make_point_id(chunk):
    """Generate deterministic IDs for source-specific chunk indices."""
    key = f"{chunk['source']}:{chunk['chunk_index']}"

    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the expanded collection if it exists.",
    )
    args = parser.parse_args()

    print("Loading documents...")
    records = load_documents()

    if not records:
        raise RuntimeError("No documents were found.")

    print(f"\nParsed records: {len(records)}")

    chunks = chunk_records(
        records,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        raise RuntimeError("No chunks were created.")

    labels = [
        (chunk["source"], int(chunk["chunk_index"]))
        for chunk in chunks
    ]

    if len(labels) != len(set(labels)):
        raise RuntimeError("Duplicate source/chunk identifiers detected.")

    sources = sorted({chunk["source"] for chunk in chunks})

    print(f"Unique documents: {len(sources)}")
    print(f"Total chunks: {len(chunks)}")

    print("\nChunk count by source:")
    for source in sources:
        count = sum(chunk["source"] == source for chunk in chunks)
        print(f"  {source}: {count}")

    # Connect to Qdrant.
    client = QdrantClient(url=QDRANT_URL)

    try:
        exists = client.collection_exists(COLLECTION_NAME)

        if exists:
            if not args.recreate:
                raise RuntimeError(
                    f"Collection '{COLLECTION_NAME}' already exists. "
                    "Use --recreate only when you intend to rebuild it."
                )

            print(f"\nDeleting existing collection: {COLLECTION_NAME}")
            client.delete_collection(COLLECTION_NAME)

        print("\nLoading embedding model...")
        model = SentenceTransformer(EMBEDDING_MODEL)

        print("Creating Qdrant collection...")

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        )

        print("Generating embeddings...")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
        )

        print("Uploading chunks...")

        for start in range(0, len(chunks), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(chunks))

            points = [
                models.PointStruct(
                    id=make_point_id(chunks[index]),
                    vector=embeddings[index].tolist(),
                    payload=chunks[index],
                )
                for index in range(start, end)
            ]

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

            print(f"Uploaded {end}/{len(chunks)} chunks")

        info = client.get_collection(COLLECTION_NAME)

        print("\n" + "=" * 60)
        print("EXPANDED INDEXING COMPLETED")
        print("=" * 60)
        print("Collection:", COLLECTION_NAME)
        print("Documents:", len(sources))
        print("Chunks uploaded:", len(chunks))
        print("Qdrant point count:", info.points_count)

    finally:
        client.close()


if __name__ == "__main__":
    main()
