import json
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.generation.answer import (
    ABSTENTION,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    MODEL_NAME,
    retrieve_context,
    generate_answer,
)


BASE_DIR = Path(__file__).resolve().parents[3]
load_dotenv(BASE_DIR / ".env")

DATASET_PATH = BASE_DIR / "data" / "evaluation" / "golden_dataset.json"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "enterprise_documents"

ABSTENTION_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "insufficient information",
]


def token_f1(expected: str, generated: str) -> float:
    """Calculate lexical token-overlap F1, not semantic correctness."""

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    expected_tokens = tokenize(expected)
    generated_tokens = tokenize(generated)

    if not expected_tokens or not generated_tokens:
        return 0.0

    expected_counts = Counter(expected_tokens)
    generated_counts = Counter(generated_tokens)
    overlap = sum((expected_counts & generated_counts).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(generated_tokens)
    recall = overlap / len(expected_tokens)

    return 2 * precision * recall / (precision + recall)


def is_abstention(answer: str) -> bool:
    """Identify an explicit insufficient-information response."""
    answer_lower = answer.lower()
    return any(
        phrase in answer_lower
        for phrase in ABSTENTION_PHRASES
    )


def main():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check your project-root .env file."
        )

    if not MODEL_NAME:
        raise RuntimeError(
            "GROQ_MODEL is missing. Check your project-root .env file."
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
    qdrant = QdrantClient(url=QDRANT_URL)
    groq_client = Groq(api_key=GROQ_API_KEY)

    answerable_count = sum(
        1 for item in dataset if item["answerable"]
    )
    unanswerable_count = len(dataset) - answerable_count

    answerable_successes = 0
    unanswerable_successes = 0
    correct_abstentions = 0
    false_answers = 0
    generation_errors = 0

    answer_f1_sum = 0.0
    answered_count = 0
    expected_source_count = 0

    try:
        for item in dataset:
            question = item["question"]
            expected_answer = item.get("expected_answer", "")
            expected_source = item.get("expected_source", "")
            answerable = item["answerable"]

            print("\n" + "=" * 65)
            print(f"Question: {question}")
            print(f"Expected answerable: {answerable}")

            answer = ""
            error = None
            context_parts = []

            try:
                context_parts = retrieve_context(
                    question,
                    embedding_model,
                    qdrant,
                )
                if "maternity leave" in question.lower():
                    print("\n--- MATERNITY RETRIEVED CONTEXT ---")
                    for part in context_parts:
                        print(part)
                    print("--- END MATERNITY RETRIEVED CONTEXT ---\n")

                if not context_parts:
                    answer = ABSTENTION
                else:
                    answer = generate_answer(
                        question,
                        context_parts,
                        groq_client,
                    )

            except Exception as exc:
                generation_errors += 1
                error = exc
                print(
                    f"Generation error: {type(exc).__name__}: {exc}"
                )

            print(
                f"Retrieved context: "
                f"{bool(context_parts) if 'context_parts' in locals() else False}"
            )
            print(f"Generated answer: {answer or 'N/A (generation failed)'}")

            # A failed generation is not a correct abstention.
            if error is not None:
                print("Evaluation status: GENERATION ERROR")
                continue

            abstained = is_abstention(answer)
            expected_source_cited = (
                bool(expected_source) and expected_source in answer
            )

            print(f"Abstained: {abstained}")

            if answerable:
                answerable_successes += 1

                if not abstained:
                    answer_f1_sum += token_f1(
                        expected_answer,
                        answer,
                    )
                    answered_count += 1

                expected_source_count += int(expected_source_cited)

                print(f"Reference answer: {expected_answer}")
                print(
                    f"Expected source cited: {expected_source_cited}"
                )

            else:
                unanswerable_successes += 1

                if abstained:
                    correct_abstentions += 1
                else:
                    false_answers += 1

                print(
                    f"Evaluation status: "
                    f"{'CORRECT ABSTENTION' if abstained else 'FALSE ANSWER'}"
                )

    finally:
        qdrant.close()
        groq_client.close()

    print("\n" + "=" * 65)
    print("LESSON 12 — ABSTENTION EVALUATION")
    print("=" * 65)
    print(f"Total questions: {len(dataset)}")
    print(f"Answerable questions in dataset: {answerable_count}")
    print(f"Unanswerable questions in dataset: {unanswerable_count}")
    print(f"Answerable generation successes: {answerable_successes}")
    print(f"Unanswerable generation successes: {unanswerable_successes}")
    print(f"Generation errors: {generation_errors}")

    if answered_count:
        print(
            "Average Answer Token F1 (answered cases): "
            f"{answer_f1_sum / answered_count:.2%}"
        )
    else:
        print("Average Answer Token F1: N/A (no answers generated)")

    if answerable_successes:
        print(
            "Expected Source Citation Rate: "
            f"{expected_source_count / answerable_successes:.2%}"
        )

    if unanswerable_successes:
        print(
            "Correct Abstention Rate: "
            f"{correct_abstentions / unanswerable_successes:.2%}"
        )
        print(
            "False Answer Rate: "
            f"{false_answers / unanswerable_successes:.2%}"
        )

    if unanswerable_count:
        print(
            "Unanswerable Generation Error Rate: "
            f"{(unanswerable_count - unanswerable_successes) / unanswerable_count:.2%}"
        )


if __name__ == "__main__":
    main()
