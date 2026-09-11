import json
import logging
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, Session, mapped_column

from .database import Base, get_db
from .dependencies import get_current_api_key
from .models import APIKey


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://host.docker.internal:11434"

# Small model used ONLY for deciding whether a chat exchange
# contains a durable memory worth storing.
#
# This is intentionally independent from the model selected by
# the user for /v1/chat/completions.
MEMORY_EXTRACTION_MODEL = "qwen3.5:0.8b"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
EMBEDDING_DIMENSIONS = 1024

# Number of relevant memories retrieved for each memory-enabled
# chat request.
MEMORY_RETRIEVAL_TOP_K = 5

# Cosine distance threshold for considering two memories duplicates.
#
# Cosine distance:
#   0.0 = identical
#   larger = less similar
#
# Kept deliberately tight so genuinely different facts are not
# accidentally merged.
MEMORY_DEDUP_DISTANCE_THRESHOLD = 0.15

# Default importance when the extraction model does not provide
# a valid importance value.
DEFAULT_MEMORY_IMPORTANCE = 5

# Maximum time allowed for Ollama memory/embedding requests.
OLLAMA_TIMEOUT = 120.0

# Keep Ollama models warm for a while after use.
OLLAMA_KEEP_ALIVE = "30m"


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

    # --------------------------------------------------------
    # Ownership
    # --------------------------------------------------------
    #
    # A memory belongs directly to an API key.
    #
    # There is intentionally NO:
    #   - conversation_id
    #   - end_user_id
    #
    # This matches the application's long-term memory design.
    #
    api_key_id: Mapped[int] = mapped_column(
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # --------------------------------------------------------
    # Memory content
    # --------------------------------------------------------

    memory: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    category: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    # Higher value = more important.
    #
    # Used when memory_limit is reached:
    #   lowest importance is removed first
    #   oldest updated memory wins ties
    #
    importance: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=DEFAULT_MEMORY_IMPORTANCE,
        server_default=str(DEFAULT_MEMORY_IMPORTANCE),
    )

    # 1024-dimensional pgvector embedding.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=True,
    )

    # --------------------------------------------------------
    # Timestamps
    # --------------------------------------------------------

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
# Access Control
# ============================================================


def require_memory_enabled(
    api_key: APIKey,
) -> None:
    """
    Require memory to be enabled for the current API key.

    Memory belongs to the API key itself, so the authenticated
    API key is the only ownership scope required.
    """

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
    """
    Generate a 1024-dimensional embedding using Ollama.
    """

    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text")

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
    """
    Retrieve the most semantically relevant memories belonging
    to the specified API key.

    IMPORTANT:
    The api_key_id filter guarantees that memories belonging to
    another API key can never be returned.
    """

    if limit <= 0:
        return []

    return (
        db.query(UserMemory)
        .filter(
            UserMemory.api_key_id == api_key_id,
            UserMemory.embedding.isnot(None),
        )
        .order_by(
            UserMemory.embedding.cosine_distance(embedding),
            UserMemory.id.asc(),
        )
        .limit(limit)
        .all()
    )


def find_similar_memory(
    db: Session,
    api_key_id: int,
    embedding: list[float],
) -> tuple[UserMemory | None, float | None]:
    """
    Find the closest existing memory for this API key.

    Returns:
        (memory, cosine_distance)

    or:

        (None, None)
    """

    row = (
        db.query(
            UserMemory,
            UserMemory.embedding.cosine_distance(embedding).label("distance"),
        )
        .filter(
            UserMemory.api_key_id == api_key_id,
            UserMemory.embedding.isnot(None),
        )
        .order_by("distance", UserMemory.id.asc())
        .first()
    )

    if row is None:
        return None, None

    memory, distance = row

    if distance is None:
        return memory, None

    return memory, float(distance)


def build_memory_context_message(
    memories: list[UserMemory],
) -> str | None:
    """
    Convert retrieved memories into a system-message block.
    """

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
# Memory Extraction
# ============================================================


_EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable, reusable facts about a user from a "
    "single chat exchange, for long-term memory storage.\n\n"
    "Rules:\n"
    "- Only extract facts that will still be true and useful in "
    "future, unrelated conversations (identity, stated preferences, "
    "stable context like their tech stack, role, or goals).\n"
    "- Do NOT extract greetings, small talk, one-off questions, "
    "temporary requests, or anything transient.\n"
    "- Do NOT extract secrets: API keys, passwords, tokens, "
    "authentication credentials, or payment/card details.\n"
    "- Do NOT store sensitive information unless it is clearly "
    "necessary and appropriate as a durable preference/context.\n"
    "- If nothing durable is present, respond with exactly: NONE\n"
    "- Otherwise respond with ONLY a compact JSON object, no other "
    "text:\n"
    '{"content": "<one durable fact, third person, e.g. '
    '"User prefers Python for backend development.">, '
    '"category": "<short category such as identity, preference, '
    'project>", '
    '"importance": <integer 1-10>}'
)


async def extract_memory_candidate(
    user_message: str,
    assistant_message: str,
) -> dict | None:
    """
    Ask the dedicated memory-extraction model whether the latest
    exchange contains a durable fact worth storing.

    Returns:

        {
            "content": "...",
            "category": "...",
            "importance": 1-10,
        }

    or None.

    Any Ollama/network exception is allowed to propagate to the
    caller. Background callers catch it so memory failures never
    break the completed chat request.
    """

    if not user_message or not user_message.strip():
        return None

    if not assistant_message or not assistant_message.strip():
        return None

    payload = {
        "model": MEMORY_EXTRACTION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": _EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"User said: {user_message}\nAssistant replied: {assistant_message}"
                ),
            },
        ],
        "stream": False,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "num_ctx": 1024,
            "num_predict": 120,
            "temperature": 0.0,
        },
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
        )

        response.raise_for_status()
        data = response.json()

    content = ((data.get("message") or {}).get("content") or "").strip()

    if not content:
        return None

    if content.upper().startswith("NONE"):
        return None

    # --------------------------------------------------------
    # Extract JSON defensively.
    #
    # Some small models may return:
    #
    # ```json
    # {...}
    # ```
    #
    # or a short sentence followed by JSON.
    # --------------------------------------------------------

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1 or end <= start:
        logger.warning("Memory extraction returned invalid JSON")
        return None

    try:
        parsed = json.loads(content[start : end + 1])
    except (ValueError, TypeError):
        logger.warning("Memory extraction JSON parsing failed")
        return None

    # --------------------------------------------------------
    # Content
    # --------------------------------------------------------

    text = str(parsed.get("content") or "").strip()

    if not text:
        return None

    # --------------------------------------------------------
    # Importance
    # --------------------------------------------------------

    importance = parsed.get(
        "importance",
        DEFAULT_MEMORY_IMPORTANCE,
    )

    try:
        importance = int(importance)
    except (TypeError, ValueError):
        importance = DEFAULT_MEMORY_IMPORTANCE

    importance = max(
        1,
        min(10, importance),
    )

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    category = parsed.get("category")

    if category is not None:
        category = str(category).strip()[:50]

        if not category:
            category = None

    # --------------------------------------------------------
    # Final candidate
    # --------------------------------------------------------

    return {
        "content": text[:4000],
        "category": category,
        "importance": importance,
    }


# ============================================================
# Memory Limit Enforcement
# ============================================================


def enforce_memory_limit(
    db: Session,
    api_key: APIKey,
) -> None:
    """
    Ensure stored memories do not exceed api_key.memory_limit.

    Eviction order:

        1. Lowest importance first
        2. Oldest updated_at first
        3. Lowest id first as deterministic tie-breaker

    memory_limit <= 0 means that no memories are retained.
    """

    limit = api_key.memory_limit or 0

    query = db.query(UserMemory).filter(UserMemory.api_key_id == api_key.id)

    count = query.count()

    overflow = count - limit

    if overflow <= 0:
        return

    victims = (
        query.order_by(
            UserMemory.importance.asc(),
            UserMemory.updated_at.asc(),
            UserMemory.id.asc(),
        )
        .limit(overflow)
        .all()
    )

    for victim in victims:
        db.delete(victim)

    db.commit()


# ============================================================
# Create / Update Memory From Chat Exchange
# ============================================================


async def remember_exchange(
    db: Session,
    api_key: APIKey,
    user_message: str,
    assistant_message: str,
) -> None:
    """
    Best-effort long-term memory pipeline.

    Steps:

        1. Check that memory is enabled/configured.
        2. Ask the extraction model whether the exchange contains
           a durable fact.
        3. Generate an embedding for the extracted fact.
        4. Find an existing near-duplicate.
        5. Update the duplicate or create a new memory.
        6. Enforce memory_limit.

    This function itself raises on upstream/database failures.
    Callers that must not break a chat response should catch the
    exception.
    """

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if not user_message or not user_message.strip():
        return

    if not assistant_message or not assistant_message.strip():
        return

    # --------------------------------------------------------
    # Memory limit
    #
    # 0 means no memories should be retained.
    # --------------------------------------------------------

    limit = api_key.memory_limit or 0

    if limit <= 0:
        return

    # --------------------------------------------------------
    # Extract durable memory candidate
    # --------------------------------------------------------

    candidate = await extract_memory_candidate(
        user_message=user_message,
        assistant_message=assistant_message,
    )

    if candidate is None:
        return

    # --------------------------------------------------------
    # Generate embedding
    # --------------------------------------------------------

    embedding = await generate_embedding(candidate["content"])

    # --------------------------------------------------------
    # Find near-duplicate
    # --------------------------------------------------------

    existing, distance = find_similar_memory(
        db=db,
        api_key_id=api_key.id,
        embedding=embedding,
    )

    is_duplicate = (
        existing is not None
        and distance is not None
        and distance <= MEMORY_DEDUP_DISTANCE_THRESHOLD
    )

    # --------------------------------------------------------
    # Update existing memory
    # --------------------------------------------------------

    if is_duplicate:
        existing.memory = candidate["content"]

        if candidate["category"]:
            existing.category = candidate["category"]

        # Never reduce the importance of an existing memory.
        existing.importance = max(
            existing.importance,
            candidate["importance"],
        )

        existing.embedding = embedding
        existing.updated_at = datetime.utcnow()

        db.commit()

    # --------------------------------------------------------
    # Create new memory
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Enforce configured record limit
    # --------------------------------------------------------

    enforce_memory_limit(
        db=db,
        api_key=api_key,
    )


# ============================================================
# Background Memory Processing
# ============================================================


async def remember_exchange_background(
    api_key_id: int,
    user_message: str,
    assistant_message: str,
) -> None:
    """
    FastAPI BackgroundTasks entry point.

    Opens a separate database session and never raises an
    exception to the client.

    This runs after the chat response has already been returned.
    """

    from .database import SessionLocal

    db = SessionLocal()

    try:
        api_key = db.query(APIKey).filter(APIKey.id == api_key_id).first()

        if api_key is None:
            logger.warning(
                "Memory processing skipped: api_key_id=%s not found",
                api_key_id,
            )
            return

        if not api_key.memory_enabled:
            return

        await remember_exchange(
            db=db,
            api_key=api_key,
            user_message=user_message,
            assistant_message=assistant_message,
        )

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
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    """
    List long-term memories belonging to the authenticated API key.
    """

    require_memory_enabled(api_key)

    query = (
        db.query(UserMemory)
        .filter(UserMemory.api_key_id == api_key.id)
        .order_by(
            UserMemory.updated_at.desc(),
            UserMemory.id.desc(),
        )
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
    """
    Delete one memory.

    The api_key_id condition is mandatory so an API key can never
    delete another API key's memory.
    """

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
# DELETE /v1/memories
# ============================================================


@router.delete("/v1/memories")
async def clear_memories(
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    """
    Delete all long-term memories belonging to the authenticated
    API key.
    """

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
