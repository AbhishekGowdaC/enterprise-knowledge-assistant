from rank_bm25 import BM25Okapi

from app.retrieval.hybrid_search import (
    COLLECTION_NAME,
    TOP_K,
    load_chunks,
    dense_search,
    bm25_search,
    exact_match_search,
    reciprocal_rank_fusion,
    tokenize,
)

from app.retrieval.reranker import CrossEncoderReranker
class RetrievalRouter:
    """
    Selects dense or hybrid retrieval.

    Hybrid chunks and the BM25 index are initialized once when
    the router is created, not rebuilt for every user question.
    """

    def __init__(
        self,
        embedding_model,
        qdrant,
        mode="dense",
        top_k=TOP_K,
        min_similarity=0.45,
    ):
        self.embedding_model = embedding_model
        self.qdrant = qdrant
        self.mode = mode.strip().lower()
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.reranker = None

        if self.mode == "hybrid":
            self.reranker = CrossEncoderReranker(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                device="cpu",
            )

        if self.mode not in ("dense", "hybrid"):
            raise ValueError(
                f"Unsupported retrieval mode: {self.mode}. "
                "Choose 'dense' or 'hybrid'."
            )

        self.chunks = None
        self.bm25 = None

        if self.mode == "hybrid":
            print("Initializing hybrid retrieval...")

            self.chunks = load_chunks(self.qdrant)

            if not self.chunks:
                raise RuntimeError(
                    f"No chunks found in Qdrant collection: "
                    f"{COLLECTION_NAME}"
                )

            tokenized_corpus = [
                tokenize(chunk["text"])
                for chunk in self.chunks
            ]

            self.bm25 = BM25Okapi(tokenized_corpus)

            print(f"Loaded {len(self.chunks)} chunks into BM25.")

    def retrieve_context(self, question):
        """Return formatted source-aware context chunks."""

        if self.mode == "dense":
            results = self._dense_retrieve(question)
        else:
            results = self._hybrid_retrieve(question)

        context_parts = []

        for item in results[:self.top_k]:
            text = item.get("text", "").strip()

            if not text:
                continue

            source = item.get("source", "unknown")
            chunk_index = item.get("chunk_index", "unknown")

            context_parts.append(
                f"[Source: {source}, chunk: {chunk_index}]\n{text}"
            )

        return context_parts

    def _dense_retrieve(self, question):
        """Use the existing dense-search configuration."""

        dense_results = dense_search(
            question,
            self.embedding_model,
            self.qdrant,
        )

        # Preserve the existing dense retrieval abstention threshold.
        if not dense_results:
            return []

        best_score = max(
            item["dense_score"] for item in dense_results
        )

        if best_score < self.min_similarity:
            return []

        return dense_results[:self.top_k]

    def _hybrid_retrieve(self, question):
        """Hybrid retrieval followed by Cross-Encoder reranking."""

        dense_results = dense_search(
            question,
            self.embedding_model,
            self.qdrant,
        )

        keyword_results = bm25_search(
            question,
            self.chunks,
            self.bm25,
        )

        exact_results = exact_match_search(
            question,
            self.chunks,
            self.bm25,
        )

        hybrid_results = reciprocal_rank_fusion(
            dense_results,
            keyword_results,
            question,
            exact_candidates=exact_results,
        )

        # Keep the top 30 candidates for reranking.
        candidates = hybrid_results[:30]

        # Rerank candidates using Cross-Encoder.
        reranked_results = self.reranker.rerank(
            question=question,
            candidates=candidates,
            top_k=self.top_k,
        )

        return reranked_results