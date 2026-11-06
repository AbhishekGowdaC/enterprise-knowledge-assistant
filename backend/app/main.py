import os
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from sqlalchemy import text
from app.database.connection import engine
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.core.logging_config import configure_logging
from fastapi import FastAPI, Request
from app.exceptions.custom_exceptions import (
    DatabaseUnavailableError,
    VectorDatabaseUnavailableError,
    LLMServiceUnavailableError,
    LLMConfigurationError,
)

from app.exceptions.handlers import (
    database_exception_handler,
    vector_database_exception_handler,
    llm_exception_handler,
    llm_configuration_exception_handler,

)

from app.generation.answer import (
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    MIN_SIMILARITY,
    MODEL_NAME,
    RETRIEVAL_MODE,
)

from app.retrieval.router import RetrievalRouter
from app.api.routes import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing from the project-root .env file.")

    if not MODEL_NAME:
        raise RuntimeError("GROQ_MODEL is missing from the project-root .env file.")

    qdrant = None
    groq_client = None

    try:
        print("Loading embedding model")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        print("Connecting to Qdrant...")
        qdrant = QdrantClient(url="http://localhost:6333")

        print("Initializing Groq client...")
        groq_client = Groq(api_key=GROQ_API_KEY)

        print(f"Initializing {RETRIEVAL_MODE.upper()} retrieval...")
        retriever = RetrievalRouter(
            embedding_model=embedding_model,
            qdrant=qdrant,
            mode=RETRIEVAL_MODE,
            top_k=5,
            min_similarity=MIN_SIMILARITY,
        )

        app.state.embedding_model = embedding_model
        app.state.qdrant = qdrant
        app.state.groq_client = groq_client
        app.state.retriever = retriever

        print("RAG resources initialized successfully.")

        yield

    finally:
        if qdrant is not None:
            qdrant.close()

        if groq_client is not None:
            groq_client.close()

        print("RAG resources closed.")
configure_logging()

app = FastAPI(
    title="Enterprise Knowledge Assistant API",
    description="Document-grounded question answering using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_exception_handler(
    DatabaseUnavailableError,
    database_exception_handler,
)

app.add_exception_handler(
    VectorDatabaseUnavailableError,
    vector_database_exception_handler,
)

app.add_exception_handler(
    LLMServiceUnavailableError,
    llm_exception_handler,
)
app.add_exception_handler(
    LLMConfigurationError,
    llm_configuration_exception_handler,
)

app.include_router(chat_router)

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.get("/ready")
def readiness_check(request: Request):
    database_status = "connected"
    vector_database_status = "connected"

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        database_status = "unavailable"

    try:
        qdrant = request.app.state.qdrant
        qdrant.get_collections()
    except Exception:
        vector_database_status = "unavailable"

    is_ready = (
        database_status == "connected"
        and vector_database_status == "connected"
    )

    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "ready" if is_ready else "not ready",
            "database": database_status,
            "vector_database": vector_database_status,
        },
    )