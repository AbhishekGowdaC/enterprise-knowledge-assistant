def chunk_records(
    records: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict]:
    """
    Split document records into overlapping character chunks.

    Chunk indices are sequential per source document rather
    than global across the complete document collection.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size"
        )

    chunks = []
    source_counters = {}

    for record in records:
        text = record["text"]
        source = record.get("source", "unknown")

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_index = source_counters.get(source, 0)

                chunks.append({
                    **record,
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                })

                source_counters[source] = chunk_index + 1

            if end == len(text):
                break

            start = end - chunk_overlap

    return chunks
