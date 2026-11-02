from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


class CrossEncoderReranker:
    """
    Reorders retrieved chunks using question-document relevance scores.

    The model scores each (question, document) pair jointly.
    Scores are raw ranking logits, not probabilities.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 8,
    ):
        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.model_name = model_name
        self.batch_size = batch_size

        print(f"Loading Cross-Encoder: {model_name}")
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )
        print(f"Cross-Encoder loaded on {device}.")

    def rerank(
        self,
        question: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Return candidates sorted by Cross-Encoder relevance score.

        Each candidate must contain a non-empty 'text' field.
        Original metadata is preserved.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not candidates:
            return []

        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate.get("text", "").strip()
        ]

        if not valid_candidates:
            return []

        pairs = [
            (question, candidate["text"])
            for candidate in valid_candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        if len(scores) != len(valid_candidates):
            raise RuntimeError(
                "Cross-Encoder returned an unexpected number of scores."
            )

        scored_candidates = []

        for candidate, score in zip(valid_candidates, scores):
            result = dict(candidate)
            result["reranker_score"] = float(score)
            scored_candidates.append(result)

        scored_candidates.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        return scored_candidates[:top_k]
