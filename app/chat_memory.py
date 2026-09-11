import json
import logging

from datetime import datetime

import httpx

from fastapi import APIRouter, Depends, HTTPException, Query

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, Session, mapped_column

from pgvector.sqlalchemy import Vector

from .database import Base, get_db
from .dependencies import get_current_api_key
from .models import APIKey

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://host.docker.internal:11434"

CHAT_MODEL = "qwen3.5:0.8b"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

EMBEDDING_DIMENSIONS = 1024

# How many relevant memories to pull into context per chat request.
MEMORY_RETRIEVAL_TOP_K = 5

# Cosine distance below this value is treated as "the same fact" for
# dedup/update purposes (0 = identical, 2 = opposite). Kept tight so
# only near-duplicate facts get merged rather than genuinely new ones.
MEMORY_DEDUP_DISTANCE_THRESHOLD = 0.15

# Default priority for new memories when the extractor omits one.
DEFAULT_MEMORY_IMPORTANCE = 5

OLLAMA_TIMEOUT = 120.0


router = APIRouter()


# ============================================================
# Database Model
# ============================================================


class UserMemory(Base):
    __tablename__ = "user_memories"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # Memory belongs directly to the API key. No conversation_id,
    # no end_user_id - this is the only ownership scope that exists.
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

    # Higher = more important. Used to decide what survives when
    # memory_limit is reached.
    importance: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=DEFAULT_MEMORY_IMPORTANCE,
        server_default=str(DEFAULT_MEMORY_IMPORTANCE),
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
# Access Control Helper
# ============================================================


def require_memory_enabled(
    api_key: APIKey,
) -> None:
    if not api_key.memory_enabled:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": "Memory is not enabled for this API key",
                    "type": "memory_access_error",
                    "code": "memory_not_enabled",
                }
            },
        )


# ============================================================
# Embedding
# ============================================================


async def generate_embedding(
    text: str,
) -> list[float]:
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
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
        raise RuntimeError("Ollama returned an empty embedding")

    embedding = embeddings[0]

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Unexpected embedding dimensions: expected "
            f"{EMBEDDING_DIMENSIONS}, got {len(embedding)}"
        )

    return embedding


# ============================================================
# Retrieval
# ============================================================


def load_relevant_memories(
    db: Session,
    api_key_id: int,
    embedding: list[float],
    limit: int = MEMORY_RETRIEVAL_TOP_K,
) -> list[UserMemory]:
    """Only ever searches memories owned by api_key_id."""

    return (
        db.query(UserMemory)
        .filter(
            UserMemory.api_key_id == api_key_id,
            UserMemory.embedding.isnot(None),
        )
        .order_by(UserMemory.embedding.cosine_distance(embedding))
        .limit(limit)
        .all()
    )


def find_similar_memory(
    db: Session,
    api_key_id: int,
    embedding: list[float],
) -> tuple[UserMemory | None, float | None]:
    """Closest existing memory for this key (if any) and its cosine distance."""

    row = (
        db.query(
            UserMemory,
            UserMemory.embedding.cosine_distance(embedding).label("distance"),
        )
        .filter(
            UserMemory.api_key_id == api_key_id,
            UserMemory.embedding.isnot(None),
        )
        .order_by("distance")
        .first()
    )

    if row is None:
        return None, None

    memory, distance = row

    return memory, distance


def build_memory_context_message(
    memories: list[UserMemory],
) -> str | None:
    """Render retrieved memories as a system-message block for the model."""

    if not memories:
        return None

    lines = "\n".join(f"- {item.memory}" for item in memories)

    return (
        "The following are known long-term facts about this user, "
        "recalled from previous conversations. Use them only if "
        "relevant to the current message; do not recite them back "
        "unless asked.\n"
        f"{lines}"
    )


# ============================================================
# Memory Extraction (what's worth remembering)
# ============================================================


_EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable, reusable facts about a user from a single "
    "chat exchange, for long-term memory storage.\n\n"
    "Rules:\n"
    "- Only extract facts that will still be true and useful in "
    "future, unrelated conversations (identity, stated preferences, "
    "stable context like their tech stack, role, or goals).\n"
    "- Do NOT extract greetings, small talk, one-off questions, or "
    "anything transient.\n"
    "- Do NOT extract secrets: API keys, passwords, tokens, or "
    "payment/card details.\n"
    "- If nothing durable is present, respond with exactly: NONE\n"
    "- Otherwise respond with ONLY a compact JSON object, no other "
    'text: {"content": "<one durable fact, third person, e.g. '
    '\'User\'s name is Santanu.\'>", "category": "<short category '
    'such as identity, preference, project>", "importance": '
    "<integer 1-10>}"
)


async def extract_memory_candidate(
    user_message: str,
    assistant_message: str,
) -> dict | None:
    """
    Ask the chat model whether the latest exchange contains a durable
    fact worth remembering. Returns {"content", "category",
    "importance"} or None. Raises on upstream/Ollama failure; callers
    are responsible for treating that as non-fatal.
    """

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User said: {user_message}\nAssistant replied: {assistant_message}"
                ),
            },
        ],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "num_ctx": 1024,
            "num_predict": 120,
            "temperature": 0.0,
        },
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

    content = ((data.get("message") or {}).get("content") or "").strip()

    if not content or content.upper().startswith("NONE"):
        return None

    # The model may wrap the JSON in prose or code fences; pull out
    # the outermost {...} block defensively.
    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(content[start : end + 1])
    except ValueError:
        return None

    text = str(parsed.get("content") or "").strip()

    if not text:
        return None

    importance = parsed.get("importance", DEFAULT_MEMORY_IMPORTANCE)

    try:
        importance = int(importance)
    except (TypeError, ValueError):
        importance = DEFAULT_MEMORY_IMPORTANCE

    importance = max(1, min(10, importance))

    category = parsed.get("category")

    if category is not None:
        category = str(category)[:50]

    return {
        "content": text[:4000],
        "category": category,
        "importance": importance,
    }


# ============================================================
# Limit Enforcement
# ============================================================


def enforce_memory_limit(
    db: Session,
    api_key: APIKey,
) -> None:
    """
    Ensure the number of stored memories for this key never exceeds
    its configured memory_limit. Evicts lowest-importance, oldest
    memories first (never a random pick). memory_limit <= 0 means no
    memories are retained at all.
    """

    limit = api_key.memory_limit or 0

    count = db.query(UserMemory).filter(UserMemory.api_key_id == api_key.id).count()

    overflow = count - limit

    if overflow <= 0:
        return

    victims = (
        db.query(UserMemory)
        .filter(UserMemory.api_key_id == api_key.id)
        .order_by(
            UserMemory.importance.asc(),
            UserMemory.updated_at.asc(),
        )
        .limit(overflow)
        .all()
    )

    for victim in victims:
        db.delete(victim)

    db.commit()


# ============================================================
# Create/Update Memory From a Chat Exchange
# ============================================================


async def remember_exchange(
    db: Session,
    api_key: APIKey,
    user_message: str,
    assistant_message: str,
) -> None:
    """
    Best-effort pipeline: decide whether the exchange contains a
    durable fact, then create or update a memory record for it and
    enforce memory_limit. Raises on failure - callers that must not
    let memory issues break a chat response should catch around this.
    """

    if not user_message or not user_message.strip():
        return

    if api_key.memory_limit is not None and api_key.memory_limit <= 0:
        return

    candidate = await extract_memory_candidate(user_message, assistant_message)

    if candidate is None:
        return

    embedding = await generate_embedding(candidate["content"])

    existing, distance = find_similar_memory(db, api_key.id, embedding)

    is_duplicate = (
        existing is not None
        and distance is not None
        and distance <= MEMORY_DEDUP_DISTANCE_THRESHOLD
    )

    if is_duplicate:
        existing.memory = candidate["content"]
        existing.category = candidate["category"] or existing.category
        existing.importance = max(existing.importance, candidate["importance"])
        existing.embedding = embedding
        existing.updated_at = datetime.utcnow()
        db.commit()
    else:
        memory = UserMemory(
            api_key_id=api_key.id,
            memory=candidate["content"],
            category=candidate["category"],
            importance=candidate["importance"],
            embedding=embedding,
        )
        db.add(memory)
        db.commit()

    enforce_memory_limit(db, api_key)


async def remember_exchange_background(
    api_key_id: int,
    user_message: str,
    assistant_message: str,
) -> None:
    """
    Self-contained entry point for use as a FastAPI background task:
    opens its own short-lived DB session and never raises, so it can
    run after the chat response has already been sent to the client.
    """

    from .database import SessionLocal

    db = SessionLocal()

    try:
        api_key = db.query(APIKey).filter(APIKey.id == api_key_id).first()

        if api_key is None or not api_key.memory_enabled:
            return

        await remember_exchange(db, api_key, user_message, assistant_message)

    except Exception:
        logger.warning(
            "Memory extraction failed for api_key_id=%s",
            api_key_id,
            exc_info=True,
        )

    finally:
        db.close()


# ============================================================
# GET /v1/memories
# ============================================================


@router.get("/v1/memories")
async def list_memories(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    require_memory_enabled(api_key)

    query = (
        db.query(UserMemory)
        .filter(UserMemory.api_key_id == api_key.id)
        .order_by(UserMemory.updated_at.desc())
    )

    total = query.count()

    memories = query.offset(offset).limit(limit).all()

    return {
        "memory_enabled": api_key.memory_enabled,
        "memory_limit": api_key.memory_limit,
        "count": total,
        "memories": [
            {
                "id": item.id,
                "content": item.memory,
                "category": item.category,
                "importance": item.importance,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in memories
        ],
    }


# ============================================================
# DELETE /v1/memories/{memory_id}
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
        "deleted": True,
        "message": "Memory deleted",
    }


# ============================================================
# DELETE /v1/memories (clear all)
# ============================================================


@router.delete("/v1/memories")
async def clear_memories(
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    require_memory_enabled(api_key)

    deleted = (
        db.query(UserMemory)
        .filter(UserMemory.api_key_id == api_key.id)
        .delete(synchronize_session=False)
    )

    db.commit()

    return {
        "deleted": deleted,
    }
