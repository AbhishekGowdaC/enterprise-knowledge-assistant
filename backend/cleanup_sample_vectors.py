
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)

COLLECTION_NAME = "enterprise_documents"
QDRANT_URL = "http://localhost:6333"

SAMPLE_SOURCES = [
    "sample.html",
    "sample_policy.pdf",
]


def main():
    client = QdrantClient(url=QDRANT_URL)

    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"Points before cleanup: {collection.points_count}")

        for source in SAMPLE_SOURCES:
            client.delete(
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
            print(f"Deleted vectors for: {source}")

        collection = client.get_collection(COLLECTION_NAME)
        print(f"Points after cleanup: {collection.points_count}")

        # Verify that no matching sample vectors remain.
        for source in SAMPLE_SOURCES:
            result = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            if result[0]:
                raise RuntimeError(
                    f"Cleanup verification failed for {source}"
                )

        print("Cleanup verification passed.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
