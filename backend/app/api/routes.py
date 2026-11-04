import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database import crud
from app.generation.answer import ABSTENTION, generate_answer
from app.schemas.chat_request import ChatRequest
from app.schemas.rag_response import RAGResponse
from app.schemas.conversation import MessageResponse, ConversationResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.post("/chat", response_model=RAGResponse)
def chat(
    payload: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Answer a question using the enterprise RAG pipeline."""

    try:
        # 1. Create or retrieve conversation
        if payload.conversation_id is None:
            conversation = crud.create_conversation(
                db,
                title=payload.question[:100],
            )
        else:
            conversation = crud.get_conversation(
                db,
                payload.conversation_id,
            )

            if conversation is None:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found.",
                )

        # 2. Save user message
        crud.create_message(
            db,
            conversation.id,
            "user",
            payload.question,
        )

        # 3. Run existing RAG pipeline
        retriever = request.app.state.retriever
        groq_client = request.app.state.groq_client

        context_parts = retriever.retrieve_context(payload.question)

        if not context_parts:
            response = RAGResponse(
                answer=ABSTENTION,
                citations=[],
                is_answerable=False,
            )
        else:
            response = generate_answer(
                payload.question,
                context_parts,
                groq_client,
            )

        # 4. Save assistant response
        crud.create_message(
            db,
            conversation.id,
            "assistant",
            response.answer,
        )

        return response

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("RAG chat request failed")

        raise HTTPException(
            status_code=502,
            detail="The RAG service could not generate a response.",
        ) from exc


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
def get_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    conversation = crud.get_conversation(db, conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    return crud.get_messages(db, conversation_id)
@router.get(
    "/conversations",
    response_model=list[ConversationResponse],
)
def list_conversations(
    db: Session = Depends(get_db),
):
    """Retrieve all conversations, newest first."""

    return crud.get_conversations(db)