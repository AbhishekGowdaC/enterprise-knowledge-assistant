from app.ingestion.parsers import parse_html
from app.ingestion.chunking import chunk_records


def test_chunking():
    records = parse_html("../data/raw/sample.html")

    chunks = chunk_records(
        records,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(chunk["text"] for chunk in chunks)
    assert all(chunk["source"] == "sample.html" for chunk in chunks)
    assert all(chunk["file_type"] == "html" for chunk in chunks)
    assert all("chunk_index" in chunk for chunk in chunks)

    print(f"Chunking test: PASS ({len(chunks)} chunks)")

    for chunk in chunks:
        print(f"\nChunk {chunk['chunk_index']}: {chunk['text']}")


if __name__ == "__main__":
    test_chunking()