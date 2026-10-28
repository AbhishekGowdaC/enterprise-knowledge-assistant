from pathlib import Path
import uuid

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)

from app.ingestion.parsers import parse_pdf, parse_html
from app.ingestion.chunking import chunk_records


# Project paths
BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data" / "raw"

# Configuration
COLLECTION_NAME = "enterprise_documents"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

from dotenv import load_dotenv

load_dotenv()

def load_documents():
    """Discover and parse all supported documents in data/raw/."""

    records = []

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    html_files = sorted(RAW_DIR.glob("*.html"))

    print(f"PDF files discovered: {len(pdf_files)}")
    print(f"HTML files discovered: {len(html_files)}")

    for file_path in pdf_files:
        print(f"Parsing PDF: {file_path.name}")
        records.extend(parse_pdf(str(file_path)))

    for file_path in html_files:
        print(f"Parsing HTML: {file_path.name}")
        records.extend(parse_html(str(file_path)))

    return records


def make_point_id(chunk):
    """Create a stable point ID for each source and chunk."""

    source = chunk["source"]
    chunk_index = chunk["chunk_index"]

    unique_key = f"{source}:{chunk_index}"

    return str(uuid.uuid5(uuid.NAMESPACE_URL, unique_key))


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {RAW_DIR}"
        )

    records = load_documents()

    if not records:
        raise ValueError(
            f"No PDF or HTML documents found in {RAW_DIR}"
        )

    print(f"\nTotal records parsed: {len(records)}")

    chunks = chunk_records(
        records,
        chunk_size=800,
        chunk_overlap=120,
    )

    if not chunks:
        raise ValueError("No chunks were created from the documents.")

    print(f"Total chunks created: {len(chunks)}")

    # Load the embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"Embedding shape: {embeddings.shape}")

    # Connect to Qdrant
    qdrant = QdrantClient(url=QDRANT_URL)

    try:
        # Check that the target collection exists.
        collections = [
            collection.name
            for collection in qdrant.get_collections().collections
        ]

        if COLLECTION_NAME not in collections:
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
                "Create it before running the indexer."
            )

        # Remove old chunks for sources being re-indexed.
        # This prevents stale chunks if a document changes.
        sources = sorted({
            chunk["source"]
            for chunk in chunks
        })

        for source in sources:
            qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="source",
                                match=MatchValue(value=source),
                            )
                        ]
                    )
                ),
                wait=True,
            )

        # Build Qdrant points
        points = []

        for chunk, embedding in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=make_point_id(chunk),
                    vector=embedding.tolist(),
                    payload=chunk,
                )
            )

        # Upload points
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        # Verify indexing
        collection_info = qdrant.get_collection(COLLECTION_NAME)

        print("\nIndexing completed successfully.")
        print(f"Sources indexed: {len(sources)}")
        print(f"Chunks uploaded: {len(points)}")
        print(f"Total points in collection: {collection_info.points_count}")
        print("Indexed source files:")

        for source in sources:
            print(f"  - {source}")

    finally:
        qdrant.close()


if __name__ == "__main__":
    main()
