import os

from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from app.exceptions.custom_exceptions import DatabaseUnavailableError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT/".env")


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not found")


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)


def get_db():
    try:
        db = SessionLocal()
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(
            "Unable to establish database session."
        ) from exc

    try:
        yield db

    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(
            "Database operation failed."
        ) from exc

    finally:
        db.close()