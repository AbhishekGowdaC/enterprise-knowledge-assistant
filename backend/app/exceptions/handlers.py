import logging

logger = logging.getLogger(__name__)

from fastapi import Request
from fastapi.responses import JSONResponse
from app.exceptions.custom_exceptions import (
    DatabaseUnavailableError,
    VectorDatabaseUnavailableError,
    LLMServiceUnavailableError,
    LLMConfigurationError,
)

async def llm_configuration_exception_handler(
    request: Request,
    exc: LLMConfigurationError,
):
    logger.exception("LLM provider configuration or authentication failed")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "AI service configuration error. Please contact support."
        },
    )

async def database_exception_handler(
    request: Request,
    exc: DatabaseUnavailableError,
):
    logger.exception("Database service is unavailable")
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Database service is temporarily unavailable."
        },
    )


async def vector_database_exception_handler(
    request: Request,
    exc: VectorDatabaseUnavailableError,
):
    logger.exception("Vector database service is unavailable")
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Document retrieval service is temporarily unavailable."
        },
    )


async def llm_exception_handler(
    request: Request,
    exc: LLMServiceUnavailableError,
):
    logger.exception("LLM provider is temporarily unavailable")
    return JSONResponse(
        status_code=502,
        content={
            "detail": "AI generation service is temporarily unavailable."
        },
    )