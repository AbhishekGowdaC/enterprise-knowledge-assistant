def chunk_records(
    records: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict]:
    """Split document records into overlapping character chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size"
        )

    chunks = []

    for record in records:
        text = record["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    **record,
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                    "start_char": start,
                    "end_char": end,
                })

            if end == len(text):
                break

            start = end - chunk_overlap

    return chunks