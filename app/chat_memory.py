import math
import uuid
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, Session, mapped_column

from .database import Base, get_db
from .dependencies import get_current_api_key


router = APIRouter(
    prefix="/v1",
    tags=["Chat Memory"],
)


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://host.docker.internal:11434"

EMBEDDING_MODEL_NAME = "qwen3-embedding:0.6b"
EMBEDDING_DIMENSIONS = 1024

OLLAMA_TIMEOUT = 300.0
OLLAMA_KEEP_ALIVE = "30m"


# ============================================================
# Database Models
# ============================================================


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )

    api_key_id: Mapped[int] = mapped_column(
        ForeignKey(
            "api_keys.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    end_user_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "conversations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    model: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    completion_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )


class UserMemory(Base):
    __tablename__ = "user_memories"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    api_key_id: Mapped[int] = mapped_column(
        ForeignKey(
            "api_keys.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    end_user_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    memory: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    category: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    # Stored as PostgreSQL vector(1024).
    # Actual column is created by SQL migration below.
    embedding = mapped_column(
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


# ============================================================
# Request Models
# ============================================================


class CreateConversationRequest(BaseModel):
    end_user_id: str = Field(
        min_length=1,
        max_length=100,
    )

    title: str | None = Field(
        default=None,
        max_length=255,
    )


class CreateMessageRequest(BaseModel):
    role: str
    content: str = Field(
        min_length=1,
        max_length=16000,
    )

    model: str | None = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CreateMemoryRequest(BaseModel):
    end_user_id: str = Field(
        min_length=1,
        max_length=100,
    )

    memory: str = Field(
        min_length=1,
        max_length=5000,
    )

    category: str | None = Field(
        default=None,
        max_length=50,
    )


# ============================================================
# Helpers
# ============================================================


def get_conversation_for_user(
    db: Session,
    api_key_id: int,
    conversation_id: uuid.UUID,
) -> Conversation:

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.api_key_id == api_key_id,
        )
        .first()
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    return conversation


async def create_embedding(
    text: str,
) -> list[float]:

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL_NAME,
                "input": text,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            },
        )

        response.raise_for_status()

        data = response.json()

    embeddings = data.get("embeddings")

    if not embeddings:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an empty embedding",
        )

    embedding = embeddings[0]

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(embedding)}"
            ),
        )

    return embedding


# ============================================================
# Conversations
# ============================================================


@router.post("/conversations")
def create_conversation(
    request: CreateConversationRequest,
    api_key=Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    conversation = Conversation(
        api_key_id=api_key.id,
        end_user_id=request.end_user_id,
        title=request.title,
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return {
        "id": str(conversation.id),
        "end_user_id": conversation.end_user_id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


@router.get("/conversations")
def list_conversations(
    end_user_id: str | None = None,
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    api_key=Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    query = db.query(Conversation).filter(Conversation.api_key_id == api_key.id)

    if end_user_id is not None:
        query = query.filter(Conversation.end_user_id == end_user_id)

    total = query.count()

    conversations = (
        query.order_by(
            Conversation.updated_at.desc(),
            Conversation.id.desc(),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "items": [
            {
                "id": str(item.id),
                "end_user_id": item.end_user_id,
                "title": item.title,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in conversations
        ],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": math.ceil(total / page_size) if total else 0,
    }


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: uuid.UUID,
    api_key=Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    conversation = get_conversation_for_user(
        db,
        api_key.id,
        conversation_id,
    )

    db.delete(conversation)
    db.commit()

    return {
        "message": "Conversation deleted",
        "id": str(conversation_id),
    }


# ============================================================
# Conversation Messages
# ============================================================


@router.post("/conversations/{conversation_id}/messages")
def create_message(
    conversation_id: uuid.UUID,
    request: CreateMessageRequest,
    api_key=Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    conversation = get_conversation_for_user(
        db,
        api_key.id,
        conversation_id,
    )

    allowed_roles = {
        "system",
        "user",
        "assistant",
        "tool",
    }

    if request.role not in allowed_roles:
        raise HTTPException(
            status_code=400,
            detail="Invalid message role",
        )

    message = ConversationMessage(
        conversation_id=conversation.id,
        role=request.role,
        content=request.content,
        model=request.model,
        prompt_tokens=request.prompt_tokens,
        completion_tokens=request.completion_tokens,
        total_tokens=request.total_tokens,
    )

    db.add(message)

    conversation.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(message)

    return {
        "id": message.id,
        "conversation_id": str(message.conversation_id),
        "role": message.role,
        "content": message.content,
        "model": message.model,
        "created_at": message.created_at,
    }


@router.get("/conversations/{conversation_id}/messages")
def list_messages(
    conversation_id: uuid.UUID,
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=50,
        ge=1,
        le=100,
    ),
    api_key=Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    conversation = get_conversation_for_user(
        db,
        api_key.id,
        conversation_id,
    )

    query = db.query(ConversationMessage).filter(
        ConversationMessage.conversation_id == conversation.id
    )

    total = query.count()

    messages = (
        query.order_by(
            ConversationMessage.created_at.asc(),
            ConversationMessage.id.asc(),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "items": [
            {
                "id": item.id,
                "role": item.role,
                "content": item.content,
                "model": item.model,
                "prompt_tokens": item.prompt_tokens,
                "completion_tokens": item.completion_tokens,
                "total_tokens": item.total_tokens,
                "created_at": item.created_at,
            }
            for item in messages
        ],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": math.ceil(total / page_size) if total else 0,
        "end_user_id": conversation.end_user_id,
    }


# ============================================================
# Long-Term Memory
# ============================================================


@router.post("/memories")
async def create_memory(
    request: CreateMemoryRequest,
    api_key=Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    embedding = await create_embedding(request.memory)

    memory = UserMemory(
        api_key_id=api_key.id,
        end_user_id=request.end_user_id,
        memory=request.memory,
        category=request.category,
        embedding=embedding,
    )

    db.add(memory)
    db.commit()
    db.refresh(memory)

    return {
        "id": memory.id,
        "end_user_id": memory.end_user_id,
        "memory": memory.memory,
        "category": memory.category,
        "created_at": memory.created_at,
    }


@router.get("/memories")
def list_memories(
    end_user_id: str,
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    api_key=Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    query = db.query(UserMemory).filter(
        UserMemory.api_key_id == api_key.id,
        UserMemory.end_user_id == end_user_id,
    )

    total = query.count()

    memories = (
        query.order_by(
            UserMemory.updated_at.desc(),
            UserMemory.id.desc(),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "items": [
            {
                "id": item.id,
                "memory": item.memory,
                "category": item.category,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in memories
        ],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": math.ceil(total / page_size) if total else 0,
    }


@router.delete("/memories/{memory_id}")
def delete_memory(
    memory_id: int,
    api_key=Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    memory = (
        db.query(UserMemory)
        .filter(
            UserMemory.id == memory_id,
            UserMemory.api_key_id == api_key.id,
        )
        .first()
    )

    if memory is None:
        raise HTTPException(
            status_code=404,
            detail="Memory not found",
        )

    db.delete(memory)
    db.commit()

    return {
        "message": "Memory deleted",
        "id": memory_id,
    }


@router.delete("/users/{end_user_id}/memories")
def delete_all_memories(
    end_user_id: str,
    api_key=Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    deleted = (
        db.query(UserMemory)
        .filter(
            UserMemory.api_key_id == api_key.id,
            UserMemory.end_user_id == end_user_id,
        )
        .delete(synchronize_session=False)
    )

    db.commit()

    return {
        "message": "All memories deleted",
        "end_user_id": end_user_id,
        "deleted": deleted,
    }
