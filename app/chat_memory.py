import uuid
from datetime import datetime

import httpx

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, Session, mapped_column
from pgvector.sqlalchemy import Vector

from .database import Base, get_db
from .dependencies import get_current_api_key
from .models import APIKey


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://host.docker.internal:11434"

CHAT_MODEL = "qwen3.5:0.8b"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

EMBEDDING_DIMENSIONS = 1024

MAX_HISTORY_MESSAGES = 10
MAX_MEMORY_RESULTS = 5


router = APIRouter()


# ============================================================
# Database Models
# ============================================================


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    api_key_id: Mapped[int] = mapped_column(
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title: Mapped[str | None] = mapped_column(
        String(200),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
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
        default=0,
        nullable=False,
    )

    completion_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )


class UserMemory(Base):
    __tablename__ = "user_memories"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    api_key_id: Mapped[int] = mapped_column(
        ForeignKey("api_keys.id", ondelete="CASCADE"),
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

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


# ============================================================
# Request Models
# ============================================================


class CreateConversationRequest(BaseModel):
    title: str | None = Field(
        default=None,
        max_length=200,
    )


class AddMessageRequest(BaseModel):
    role: str
    content: str
    model: str | None = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CreateMemoryRequest(BaseModel):
    memory: str = Field(
        min_length=1,
        max_length=4000,
    )

    category: str | None = Field(
        default=None,
        max_length=50,
    )


# ============================================================
# Helpers
# ============================================================


def get_owned_conversation(
    db: Session,
    api_key_id: int,
    conversation_id: str,
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


def require_memory_enabled(
    api_key: APIKey,
) -> None:

    if not api_key.memory_enabled:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": ("Memory is not enabled for this API key"),
                    "type": "memory_access_error",
                    "code": "memory_not_enabled",
                }
            },
        )


async def generate_embedding(
    text: str,
) -> list[float]:

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": text,
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
            detail={
                "error": "Invalid embedding dimensions",
                "expected": EMBEDDING_DIMENSIONS,
                "actual": len(embedding),
            },
        )

    return embedding


def load_recent_messages(
    db: Session,
    conversation_id: str,
    limit: int = MAX_HISTORY_MESSAGES,
) -> list[ConversationMessage]:

    rows = (
        db.query(ConversationMessage)
        .filter(ConversationMessage.conversation_id == conversation_id)
        .order_by(
            ConversationMessage.created_at.desc(),
            ConversationMessage.id.desc(),
        )
        .limit(limit)
        .all()
    )

    rows.reverse()

    return rows


def load_relevant_memories(
    db: Session,
    api_key_id: int,
    embedding: list[float],
    limit: int = MAX_MEMORY_RESULTS,
) -> list[UserMemory]:

    memories = (
        db.query(UserMemory)
        .filter(
            UserMemory.api_key_id == api_key_id,
            UserMemory.embedding.isnot(None),
        )
        .order_by(UserMemory.embedding.cosine_distance(embedding))
        .limit(limit)
        .all()
    )

    return memories


async def save_memory(
    db: Session,
    api_key_id: int,
    memory_text: str,
    category: str | None = None,
) -> UserMemory:

    embedding = await generate_embedding(memory_text)

    memory = UserMemory(
        api_key_id=api_key_id,
        memory=memory_text,
        category=category,
        embedding=embedding,
    )

    db.add(memory)
    db.commit()
    db.refresh(memory)

    return memory


# ============================================================
# Create Conversation
# ============================================================


@router.post("/v1/conversations")
async def create_conversation(
    request: CreateConversationRequest,
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    conversation = Conversation(
        id=str(uuid.uuid4()),
        api_key_id=api_key.id,
        title=request.title,
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


# ============================================================
# List Conversations
# ============================================================


@router.get("/v1/conversations")
async def list_conversations(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    query = (
        db.query(Conversation)
        .filter(Conversation.api_key_id == api_key.id)
        .order_by(Conversation.updated_at.desc())
    )

    total = query.count()

    conversations = query.offset((page - 1) * page_size).limit(page_size).all()

    total_pages = (total + page_size - 1) // page_size if total else 1

    return {
        "items": [
            {
                "id": item.id,
                "title": item.title,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in conversations
        ],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
    }


# ============================================================
# Conversation Messages
# ============================================================


@router.get("/v1/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    conversation = get_owned_conversation(
        db=db,
        api_key_id=api_key.id,
        conversation_id=conversation_id,
    )

    query = (
        db.query(ConversationMessage)
        .filter(ConversationMessage.conversation_id == conversation.id)
        .order_by(
            ConversationMessage.created_at.asc(),
            ConversationMessage.id.asc(),
        )
    )

    total = query.count()

    messages = query.offset((page - 1) * page_size).limit(page_size).all()

    total_pages = (total + page_size - 1) // page_size if total else 1

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
        "total_pages": total_pages,
    }


# ============================================================
# Add Message Manually
# ============================================================


@router.post("/v1/conversations/{conversation_id}/messages")
async def add_message(
    conversation_id: str,
    request: AddMessageRequest,
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    conversation = get_owned_conversation(
        db=db,
        api_key_id=api_key.id,
        conversation_id=conversation_id,
    )

    if request.role not in {
        "system",
        "user",
        "assistant",
        "tool",
    }:
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
        "conversation_id": message.conversation_id,
        "role": message.role,
        "content": message.content,
        "model": message.model,
        "created_at": message.created_at,
    }


# ============================================================
# Create Long-Term Memory
# ============================================================


@router.post("/v1/memories")
async def create_memory(
    request: CreateMemoryRequest,
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    require_memory_enabled(api_key)

    memory = await save_memory(
        db=db,
        api_key_id=api_key.id,
        memory_text=request.memory,
        category=request.category,
    )

    return {
        "id": memory.id,
        "memory": memory.memory,
        "category": memory.category,
        "created_at": memory.created_at,
    }


# ============================================================
# List Memories
# ============================================================


@router.get("/v1/memories")
async def list_memories(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    require_memory_enabled(api_key)

    query = (
        db.query(UserMemory)
        .filter(UserMemory.api_key_id == api_key.id)
        .order_by(UserMemory.created_at.desc())
    )

    total = query.count()

    memories = query.offset((page - 1) * page_size).limit(page_size).all()

    total_pages = (total + page_size - 1) // page_size if total else 1

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
        "total_pages": total_pages,
    }


# ============================================================
# Delete Memory
# ============================================================


@router.delete("/v1/memories/{memory_id}")
async def delete_memory(
    memory_id: int,
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    require_memory_enabled(api_key)

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
        "id": memory_id,
        "message": "Memory deleted",
    }
