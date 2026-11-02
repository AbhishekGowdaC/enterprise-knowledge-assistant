import csv
import json
import re
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
    generate_answer,
)
from app.retrieval.router import RetrievalRouter


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]
load_dotenv(BASE_DIR / ".env")

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "final"
    / "expanded_golden_dataset_v1.json"
)

RESULTS_DIR = BASE_DIR / "data" / "evaluation" / "results"

QDRANT_URL = "http://localhost:6333"
RETRIEVAL_MODE = "hybrid"
TOP_K = 5


# --------------------------------------------------
# 2. Evaluation helpers
# --------------------------------------------------

def normalize_text(text):
    return re.findall(r"\b\w+\b", text.lower())


def token_f1(prediction, expected):
    predicted_tokens = normalize_text(prediction)
    expected_tokens = normalize_text(expected)

    if not predicted_tokens or not expected_tokens:
        return 0.0

    common = {}

    for token in predicted_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0

    for token in expected_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)

    return (
        2 * precision * recall
        / (precision + recall)
    )


def has_expected_chunk(question, context_parts):
    expected_chunks = question.get("relevant_chunks", [])

    for expected in expected_chunks:
        source = expected["source"]
        chunk_index = str(expected["chunk_index"])

        prefix = f"[Source: {source}, chunk: {chunk_index}]"

        for context in context_parts:
            if context.startswith(prefix):
                return True

    return False


def citation_count(answer):
    pattern = r"\[Source:\s*.+?,\s*chunk:\s*\d+\]"
    return len(re.findall(pattern, answer))


# --------------------------------------------------
# 3. Main evaluation
# --------------------------------------------------

def main():

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing.")

    if not MODEL_NAME:
        raise RuntimeError("GROQ_MODEL is missing.")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    questions = dataset["questions"]

    print("=" * 65)
    print("FULL RAG PIPELINE EVALUATION")
    print("=" * 65)
    print(f"Dataset: {DATASET_PATH.name}")
    print(f"Total questions: {len(questions)}")
    print(
        "Answerable:",
        sum(q["answerable"] for q in questions),
    )
    print(
        "Unanswerable:",
        sum(not q["answerable"] for q in questions),
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    qdrant = QdrantClient(url=QDRANT_URL)
    groq_client = Groq(api_key=GROQ_API_KEY)

    results = []

    try:
        print("\nInitializing retrieval router...")

        retriever = RetrievalRouter(
            embedding_model=embedding_model,
            qdrant=qdrant,
            mode=RETRIEVAL_MODE,
            top_k=TOP_K,
        )

        print("\nStarting evaluation...\n")

        for index, question_data in enumerate(questions, start=1):

            question_id = question_data["id"]
            question = question_data["question"]
            answerable = question_data["answerable"]

            expected_answer = question_data.get(
                "expected_answer", ""
            )

            print(
                f"[{index}/{len(questions)}] "
                f"{question_id}: {question}"
            )

            result = {
                "id": question_id,
                "question": question,
                "answerable": answerable,
                "retrieval_hit": False,
                "generated_answer": "",
                "answer_f1": 0.0,
                "correct_abstention": False,
                "citation_count": 0,
                "generation_error": "",
            }

            try:
                # Step 1: Retrieve context using the real application router.
                context_parts = retriever.retrieve_context(question)

                # Step 2: Check whether expected evidence was retrieved.
                result["retrieval_hit"] = has_expected_chunk(
                    question_data,
                    context_parts,
                )

                # Step 3: Generate an answer or abstain if context is empty.
                if not context_parts:
                    answer = ABSTENTION
                else:
                    answer = generate_answer(
                        question,
                        context_parts,
                        groq_client,
                    )

                result["generated_answer"] = answer

                is_abstention = answer.strip() == ABSTENTION

                if answerable:
                    if not is_abstention:
                        result["answer_f1"] = token_f1(
                            answer,
                            expected_answer,
                        )
                else:
                    result["correct_abstention"] = is_abstention

                result["citation_count"] = citation_count(answer)

                status = (
                    "ABSTAINED"
                    if is_abstention
                    else "ANSWERED"
                )

                print(f"    Status: {status}")
                print(
                    f"    Retrieval hit: "
                    f"{result['retrieval_hit']}"
                )
                print(
                    f"    Answer F1: "
                    f"{result['answer_f1']:.4f}"
                )

            except Exception as error:
                result["generation_error"] = str(error)
                print(f"    ERROR: {error}")

            results.append(result)

    finally:
        qdrant.close()
        groq_client.close()

    # --------------------------------------------------
    # 4. Calculate metrics
    # --------------------------------------------------

    answerable_results = [
        r for r in results if r["answerable"]
    ]

    unanswerable_results = [
        r for r in results if not r["answerable"]
    ]

    def safe_rate(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    retrieval_hits = sum(
        r["retrieval_hit"] for r in answerable_results
    )

    answered = sum(
        bool(r["generated_answer"])
        and r["generated_answer"] != ABSTENTION
        and not r["generation_error"]
        for r in answerable_results
    )

    correct_abstentions = sum(
        r["correct_abstention"]
        for r in unanswerable_results
    )

    average_f1 = (
        sum(r["answer_f1"] for r in answerable_results)
        / len(answerable_results)
        if answerable_results
        else 0.0
    )
    errors = sum(
        bool(r["generation_error"])
        for r in results
    )

    summary = {
        "total_questions": len(results),
        "answerable_questions": len(answerable_results),
        "unanswerable_questions": len(unanswerable_results),
        "retrieval_hits": retrieval_hits,
        "retrieval_hit_rate": safe_rate(
            retrieval_hits,
            len(answerable_results),
        ),
        "answered_answerable_questions": answered,
        "answer_rate": safe_rate(
            answered,
            len(answerable_results),
        ),
        "average_token_f1_nonzero": average_f1,
        "correct_abstentions": correct_abstentions,
        "unanswerable_abstention_rate": safe_rate(
            correct_abstentions,
            len(unanswerable_results),
        ),
        "generation_errors": errors,
    }

    # --------------------------------------------------
    # 5. Save results
    # --------------------------------------------------

    csv_path = RESULTS_DIR / "full_rag_evaluation.csv"
    json_path = RESULTS_DIR / "full_rag_summary.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # --------------------------------------------------
    # 6. Print summary
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("FULL RAG EVALUATION SUMMARY")
    print("=" * 65)

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nResults saved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
