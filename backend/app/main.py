import os
from contextlib import asynccontextmanager


from dotenv import load_dotenv
from fastapi import FastAPI
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

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

app = FastAPI(
    title="Enterprise Knowledge Assistant API",
    description="Document-grounded question answering using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(chat_router)
