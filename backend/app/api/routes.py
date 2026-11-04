import logging

from fastapi import APIRouter, HTTPException, Request

from app.generation.answer import (
    ABSTENTION,
    generate_answer,
)
from app.schemas.chat_request import ChatRequest
from app.schemas.rag_response import RAGResponse


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.post("/chat", response_model=RAGResponse)
def chat(payload: ChatRequest, request: Request):
    """Answer a question using the enterprise RAG pipeline."""

    try:
        retriever = request.app.state.retriever
        groq_client = request.app.state.groq_client

        context_parts = retriever.retrieve_context(payload.question)

        if not context_parts:
            return RAGResponse(
                answer=ABSTENTION,
                citations=[],
                is_answerable=False,
            )

        response = generate_answer(
            payload.question,
            context_parts,
            groq_client,
        )

        return response

    except Exception as exc:
        logger.exception("RAG chat request failed")

        raise HTTPException(
            status_code=502,
            detail="The RAG service could not generate a response.",
        ) from exc
