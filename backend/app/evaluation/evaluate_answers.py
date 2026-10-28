
import json
import re
from collections import Counter
from pathlib import Path

from google import genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.generation.answer import (
    API_KEY,
    EMBEDDING_MODEL,
    retrieve_context,
    generate_answer,
)


BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_PATH = BASE_DIR / "data" / "evaluation" / "golden_dataset.json"

COLLECTION_NAME = "enterprise_documents"

ABSTENTION_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "insufficient information",
]


def token_f1(expected: str, generated: str) -> float:
    """
    Calculate token-overlap F1 between the reference answer
    and generated answer.

    This is a lexical metric, not a semantic correctness metric.
    """

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    expected_tokens = tokenize(expected)
    generated_tokens = tokenize(generated)

    if not expected_tokens or not generated_tokens:
        return 0.0

    expected_counts = Counter(expected_tokens)
    generated_counts = Counter(generated_tokens)

    overlap = sum(
        (expected_counts & generated_counts).values()
    )

    if overlap == 0:
        return 0.0

    precision = overlap / len(generated_tokens)
    recall = overlap / len(expected_tokens)

    return 2 * precision * recall / (precision + recall)


def is_abstention(answer: str) -> bool:
    """Check whether the answer uses a recognized abstention phrase."""

    answer_lower = answer.lower()

    return any(
        phrase in answer_lower
        for phrase in ABSTENTION_PHRASES
    )


def main():
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Check your project-root .env file."
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("The evaluation dataset is empty.")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    qdrant = QdrantClient(url="http://localhost:6333")
    gemini = genai.Client(api_key=API_KEY)

    answerable_count = 0
    answer_f1_sum = 0.0
    citation_count = 0

    unanswerable_count = 0
    correct_abstention_count = 0

    try:
        for item in dataset:
            question = item["question"]
            expected_answer = item["expected_answer"]
            expected_source = item["expected_source"]
            answerable = item["answerable"]

            # 1. Retrieve relevant document chunks.
            context_parts = retrieve_context(
                question,
                embedding_model,
                qdrant,
            )

            # 2. Generate an answer or abstain if retrieval
            #    returns no usable context.
            if context_parts:
                answer = generate_answer(
                    question,
                    context_parts,
                    gemini,
                )
            else:
                answer = (
                    "I don't have enough relevant evidence "
                    "in the available documents to answer "
                    "that question."
                )

            # 3. Evaluate abstention behavior.
            abstained = is_abstention(answer)

            # 4. Evaluate lexical overlap for answerable questions.
            answer_score = (
                token_f1(expected_answer, answer)
                if answerable and expected_answer
                else 0.0
            )

            # 5. Check whether the expected source filename appears
            #    in the generated answer.
            cited_expected_source = (
                bool(expected_source)
                and expected_source in answer
            )

            if answerable:
                answerable_count += 1
                answer_f1_sum += answer_score
                citation_count += int(cited_expected_source)
            else:
                unanswerable_count += 1
                correct_abstention_count += int(abstained)

            # 6. Print per-question results.
            print("\n" + "=" * 65)
            print(f"Question: {question}")
            print(f"Answerable: {answerable}")
            print(
                "Retrieved context available: "
                f"{bool(context_parts)}"
            )
            print(f"Generated answer: {answer}")
            print(f"Abstained: {abstained}")

            if answerable:
                print(f"Expected answer: {expected_answer}")
                print(f"Answer token F1: {answer_score:.2%}")
                print(
                    "Expected source cited: "
                    f"{cited_expected_source}"
                )

    finally:
        qdrant.close()
        gemini.close()

    # 7. Print aggregate metrics.
    print("\n" + "=" * 65)
    print("ANSWER EVALUATION SUMMARY")
    print("=" * 65)

    print(f"Total questions: {len(dataset)}")
    print(f"Answerable questions: {answerable_count}")
    print(f"Unanswerable questions: {unanswerable_count}")

    if answerable_count:
        print(
            "Average Answer Token F1: "
            f"{answer_f1_sum / answerable_count:.2%}"
        )

        print(
            "Expected Source Citation Rate: "
            f"{citation_count / answerable_count:.2%}"
        )

    if unanswerable_count:
        print(
            "Correct Abstention Rate: "
            f"{correct_abstention_count / unanswerable_count:.2%}"
        )


if __name__ == "__main__":
    main()
