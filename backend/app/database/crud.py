from sqlalchemy.orm import Session
from app.database.models import Conversation, Message
from sqlalchemy import select

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
    except Exception:
        db.rollback()
        raise


def get_conversation(
        db: Session,
        converation_id: int,
) -> Conversation | None:

    return db.get(Conversation, converation_id)



def create_message(
        db:Session,
        conversation_id: int,
        role: str,
        content: str
) -> Message:
    try:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content
        )
        db.add(message)
        db.commit()
        db.refresh(message)

        return message

    except Exception:
        db.rollback()
        raise

def get_messages(
        db: Session,
        conversation_id: int,
) -> list[Message]:

    statement = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    )

    return list(db.scalars(statement).all())


def get_conversations(db: Session) -> list[Conversation]:
    statement = (
        select(Conversation)
        .order_by(Conversation.created_at.desc(), Conversation.id.desc())
    )

    return list(db.scalars(statement).all())


def delete_conversation(
        db: Session,
        conversation_id: int
) -> bool:
    conversation = get_conversation(db, conversation_id)

    if conversation is None:
        return False

    db.delete(conversation)
    db.commit()
    return True


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

    except Exception:
        db.rollback()
        raise