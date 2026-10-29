import re

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "enterprise_documents"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 5
CANDIDATE_K = 30          # per-retriever candidate pool (was 10)
RRF_K = 60

# A chunk ranked #1 in both lists scores ~0.033 under RRF (k=60), so the
# bonuses are kept small relative to that: they nudge, not override.
EXACT_MATCH_BONUS = 0.008
HOTLINE_CONTEXT_BONUS = 0.003

# Ignore very short numeric tokens ("2", "10") which match too much noise.
MIN_EXACT_TOKEN_LEN = 3

# Words and hyphenated identifiers, e.g. "inc-4521" stays whole.
TOKEN_RE = re.compile(r"\b\w+(?:-\w+)*\b")

# Word-boundary match, so "call" no longer hits "recall" or "calculate".
HOTLINE_RE = re.compile(
    r"\b(?:hotline|extension|security incidents?|phishing|calls?)\b"
)


def tokenize(text: str) -> list[str]:
    """
    Lowercase word tokens for BM25.

    Hyphenated identifiers are indexed both whole ("inc-4521") and
    as parts ("inc", "4521"), so exact and partial queries both match.
    """
    tokens = []

    for token in TOKEN_RE.findall(text.lower()):
        tokens.append(token)

        if "-" in token:
            tokens.extend(token.split("-"))

    return tokens


def extract_exact_tokens(question: str) -> list[str]:
    """Return numeric/identifier tokens (contain a digit) from the query."""
    tokens = TOKEN_RE.findall(question.lower())

    return [
        token
        for token in dict.fromkeys(tokens)  # dedupe, keep order
        if any(char.isdigit() for char in token)
        and len(token) >= MIN_EXACT_TOKEN_LEN
    ]


def compile_exact_patterns(tokens: list[str]) -> list[re.Pattern]:
    """Whole-token regexes so '4521' does not match '14521'."""
    return [
        re.compile(rf"(?<!\w){re.escape(token)}(?!\w)")
        for token in tokens
    ]


def load_chunks(client: QdrantClient) -> list[dict]:
    """Load indexed document chunks and metadata from Qdrant."""
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
                chunks.append({
                    "id": str(point.id),
                    "text": text,
                    "source": payload.get("source", "unknown"),
                    "chunk_index": payload.get("chunk_index", -1),
                })

        if next_offset is None:
            break

        offset = next_offset

    return chunks


def dense_search(
    question: str,
    model: SentenceTransformer,
    client: QdrantClient,
) -> list[dict]:
    """Retrieve chunks using semantic embedding similarity."""
    query_vector = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=CANDIDATE_K,
        with_payload=True,
    )

    results = []

    for point in response.points:
        payload = point.payload or {}
        text = payload.get("text", "").strip()

        if not text:
            continue

        results.append({
            "id": str(point.id),
            "text": text,
            "source": payload.get("source", "unknown"),
            "chunk_index": payload.get("chunk_index", -1),
            "dense_score": float(point.score),
        })

    return results


def bm25_search(
    question: str,
    chunks: list[dict],
    bm25: BM25Okapi,
) -> list[dict]:
    """Retrieve chunks using keyword matching."""
    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:CANDIDATE_K]

    results = []

    for index in ranked_indices:
        results.append({
            **chunks[index],
            "bm25_score": float(scores[index]),
        })

    return results


def exact_match_search(
    question: str,
    chunks: list[dict],
    bm25: BM25Okapi,
) -> list[dict]:
    """
    Find chunks containing the query's exact numbers/identifiers,
    even if neither retriever ranked them in its top-K.
    Matches are ordered by BM25 score and capped at CANDIDATE_K.
    """
    tokens = extract_exact_tokens(question)

    if not tokens:
        return []

    patterns = compile_exact_patterns(tokens)

    matches = [
        index
        for index, chunk in enumerate(chunks)
        if any(p.search(chunk["text"].lower()) for p in patterns)
    ]

    if not matches:
        return []

    scores = bm25.get_scores(tokenize(question))
    matches.sort(key=lambda i: scores[i], reverse=True)

    return [
        {**chunks[i], "bm25_score": float(scores[i])}
        for i in matches[:CANDIDATE_K]
    ]


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    question: str = "",
    exact_candidates: list[dict] | None = None,
) -> list[dict]:
    """
    Fuse dense and BM25 rankings using RRF.

    Chunks that contain exact numbers/identifiers from the question get a
    small bonus. Those chunks are also injected into the candidate pool
    (with no RRF contribution of their own) so they are never missed just
    because they fell outside both top-K lists.

    Chunks that also use hotline-style terminology get a further small
    bonus. This is a domain heuristic, not a guarantee of relevance.
    """
    fused = {}

    def new_entry(result: dict) -> dict:
        return {
            **result,
            "rrf_score": 0.0,
            "dense_rank": None,
            "bm25_rank": None,
            "exact_match_bonus": 0.0,
            "hotline_context_bonus": 0.0,
        }

    ranked_lists = [
        ("dense_rank", dense_results),
        ("bm25_rank", bm25_results),
    ]

    # Step 1: Calculate the RRF score.
    for rank_name, results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]

            if chunk_id not in fused:
                fused[chunk_id] = new_entry(result)

            fused[chunk_id]["rrf_score"] += 1.0 / (RRF_K + rank)
            fused[chunk_id][rank_name] = rank

            if "dense_score" in result:
                fused[chunk_id]["dense_score"] = result["dense_score"]

            if "bm25_score" in result:
                fused[chunk_id]["bm25_score"] = result["bm25_score"]

    # Step 2: Inject exact-match chunks missed by both retrievers.
    for result in exact_candidates or []:
        if result["id"] not in fused:
            fused[result["id"]] = new_entry(result)

    # Step 3: Find exact numbers and identifiers in the question.
    patterns = compile_exact_patterns(extract_exact_tokens(question))

    # Step 4: Apply exact-token and contextual bonuses.
    if patterns:
        for item in fused.values():
            text = item.get("text", "").lower()

            if not any(p.search(text) for p in patterns):
                continue

            item["exact_match_bonus"] = EXACT_MATCH_BONUS
            item["rrf_score"] += EXACT_MATCH_BONUS

            if HOTLINE_RE.search(text):
                item["hotline_context_bonus"] = HOTLINE_CONTEXT_BONUS
                item["rrf_score"] += HOTLINE_CONTEXT_BONUS

    # Step 5: Return the highest-scoring chunks first.
    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


def print_ranked(title: str, results: list[dict], score_key: str) -> None:
    print(f"\n--- {title} ---")

    for rank, result in enumerate(results[:3], start=1):
        print(
            f"{rank}. {result['source']} | "
            f"chunk {result['chunk_index']} | "
            f"score={result[score_key]:.4f}"
        )


def main():
    client = QdrantClient(url=QDRANT_URL)

    try:
        chunks = load_chunks(client)

        if not chunks:
            raise RuntimeError(
                "No indexed chunks found. Run document indexing first."
            )

        print(f"Loaded chunks: {len(chunks)}")
        print(f"Loading embedding model: {EMBEDDING_MODEL}")

        model = SentenceTransformer(EMBEDDING_MODEL)

        bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

        print("Hybrid search is ready.")

        while True:
            try:
                question = input(
                    "\nEnter a question (or type 'exit'): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if question.lower() == "exit":
                break

            if not question:
                print("Please enter a question.")
                continue

            dense_results = dense_search(question, model, client)
            keyword_results = bm25_search(question, chunks, bm25)
            exact_results = exact_match_search(question, chunks, bm25)

            fused_results = reciprocal_rank_fusion(
                dense_results,
                keyword_results,
                question,
                exact_candidates=exact_results,
            )

            print_ranked("Dense search top 3", dense_results, "dense_score")
            print_ranked("BM25 search top 3", keyword_results, "bm25_score")

            print("\n--- Hybrid search top 5 ---")

            for rank, result in enumerate(fused_results[:TOP_K], start=1):
                print(f"\nRank: {rank}")
                print(f"RRF score: {result['rrf_score']:.5f}")
                print(f"Source: {result['source']}")
                print(f"Chunk: {result['chunk_index']}")
                print(f"Dense rank: {result['dense_rank']}")
                print(f"BM25 rank: {result['bm25_rank']}")
                print(
                    "Exact-match bonus: "
                    f"{result['exact_match_bonus']:.3f}"
                )
                print(
                    "Hotline-context bonus: "
                    f"{result['hotline_context_bonus']:.3f}"
                )
                print(f"Text: {result['text']!r}")

    finally:
        client.close()


if __name__ == "__main__":
    main()