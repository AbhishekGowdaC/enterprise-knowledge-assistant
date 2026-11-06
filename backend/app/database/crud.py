from sqlalchemy.orm import Session
from app.database.models import Conversation, Message
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from app.exceptions.custom_exceptions import DatabaseUnavailableError

def create_conversation(
    db: Session,
    title: str,
) -> Conversation:
    try:
        conversation = Conversation(title=title)

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return conversation

    except OperationalError as exc:
        db.rollback()

        raise DatabaseUnavailableError(
            "Failed to create conversation."
        ) from exc

    except Exception:
        db.rollback()
        raise

def get_conversation(
    db: Session,
    conversation_id: int,
) -> Conversation | None:
    try:
        return db.get(Conversation, conversation_id)

    except OperationalError as exc:
        raise DatabaseUnavailableError(
            "Failed to retrieve conversation."
        ) from exc

def create_message(
    db: Session,
    conversation_id: int,
    role: str,
    content: str,
) -> Message:
    try:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )

        db.add(message)
        db.commit()
        db.refresh(message)

        return message

    except OperationalError as exc:
        db.rollback()

        raise DatabaseUnavailableError(
            "Failed to create message."
        ) from exc

    except Exception:
        db.rollback()
        raise
        
def get_messages(
    db: Session,
    conversation_id: int,
) -> list[Message]:
    try:
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )

        return list(db.scalars(statement).all())

    except OperationalError as exc:
        raise DatabaseUnavailableError(
            "Failed to retrieve messages."
        ) from exc

def get_conversations(db: Session) -> list[Conversation]:
    try:
        statement = (
            select(Conversation)
            .order_by(
                Conversation.created_at.desc(),
                Conversation.id.desc()
            )
        )

        return list(db.scalars(statement).all())

    except OperationalError as exc:
        raise DatabaseUnavailableError(
            "Failed to retrieve conversations."
        ) from exc

    
def delete_conversation(
    db: Session,
    conversation_id: int,
) -> bool:
    try:
        conversation = get_conversation(db, conversation_id)

        if conversation is None:
            return False

        db.delete(conversation)
        db.commit()

        return True

    except OperationalError as exc:
        db.rollback()

        raise DatabaseUnavailableError(
            "Failed to delete conversation."
        ) from exc

    except Exception:
        db.rollback()
        raise


def update_conversation_title(
    db: Session,
    conversation_id: int,
    new_title: str,
) -> Conversation | None:
    try:
        conversation = get_conversation(db, conversation_id)

        if conversation is None:
            return None

        conversation.title = new_title

        db.commit()
        db.refresh(conversation)

        return conversation

    except OperationalError as exc:
        db.rollback()

        raise DatabaseUnavailableError(
            "Failed to update conversation title."
        ) from exc

    except Exception:
        db.rollback()
        raise