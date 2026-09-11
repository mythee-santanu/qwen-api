import json
import logging
import os
import secrets
import time
import uuid

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import AsyncGenerator, Literal

import httpx

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    status,
)

from fastapi.responses import StreamingResponse

from pydantic import BaseModel, Field

from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .dependencies import get_current_api_key
from .models import APIKey, APIKeyModelAccess, APIUsage

from .chat_memory import (
    router as chat_memory_router,
    build_memory_context_message,
    generate_embedding,
    load_relevant_memories,
    remember_exchange,
    remember_exchange_background,
)

from .security import (
    generate_api_key,
    get_key_prefix,
    hash_api_key,
)

logger = logging.getLogger(__name__)


# ============================================================
# Database
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Qwen API",
    version="3.0.0",
    description="OpenAI-compatible local Qwen API",
)

app.include_router(chat_memory_router)
# ============================================================
# Ollama Configuration
# ============================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://host.docker.internal:11434",
)

MODEL_NAME = "qwen3.5:0.8b"

EMBEDDING_MODEL_NAME = "qwen3-embedding:0.6b"

EMBEDDING_DIMENSIONS = 1024


# ============================================================
# Resource Limits
# ============================================================

# Chat
MAX_MESSAGES = 32
MAX_MESSAGE_CHARS = 16000
MAX_TOTAL_INPUT_CHARS = 50000
MAX_OUTPUT_TOKENS = 512

# Embeddings
MAX_EMBEDDING_BATCH = 8
MAX_EMBEDDING_CHARS = 16000
MAX_TOTAL_EMBEDDING_CHARS = 50000

# Default limits for newly created keys
DEFAULT_REQUESTS_PER_MINUTE = 30
DEFAULT_DAILY_TOKEN_LIMIT = 10000
DEFAULT_MONTHLY_TOKEN_LIMIT = 500000

# Ollama
OLLAMA_TIMEOUT = 300.0
OLLAMA_KEEP_ALIVE = "30m"

# Approximate token estimation for quota pre-checks.
# Actual chat token usage comes from Ollama.
CHARS_PER_TOKEN_ESTIMATE = 4


# ============================================================
# Admin
# ============================================================

ADMIN_MASTER_KEY = os.getenv("ADMIN_MASTER_KEY")

if not ADMIN_MASTER_KEY:
    raise RuntimeError("ADMIN_MASTER_KEY is not configured")


# ============================================================
# Supported Models
# ============================================================

SUPPORTED_MODELS = {
    MODEL_NAME,
    EMBEDDING_MODEL_NAME,
}


# ============================================================
# In-memory rate limiter
# ============================================================

_rate_limit_store: dict[
    tuple[int, str],
    deque[float],
] = defaultdict(deque)


# ============================================================
# Request Models
# ============================================================

LimitValue = int | Literal["unlimited"]


class ModelLimitRequest(BaseModel):
    enabled: bool = True

    requests_per_minute: LimitValue = DEFAULT_REQUESTS_PER_MINUTE

    daily_token_limit: LimitValue = DEFAULT_DAILY_TOKEN_LIMIT

    monthly_token_limit: LimitValue = DEFAULT_MONTHLY_TOKEN_LIMIT


class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    memory_enabled: bool = False
    memory_limit: int = Field(default=0, ge=0)
    models: dict[str, ModelLimitRequest] = {}


class MemorySettingsUpdateRequest(BaseModel):
    memory_enabled: bool | None = None
    memory_limit: int | None = Field(default=None, ge=0)


class QuotaUpdateRequest(BaseModel):
    models: dict[str, ModelLimitRequest]


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str

    messages: list[Message]

    stream: bool = False

    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
    )

    max_tokens: int = Field(
        default=256,
        ge=1,
        le=MAX_OUTPUT_TOKENS,
    )


class EmbeddingRequest(BaseModel):
    model: str

    input: str | list[str]


# ============================================================
# Admin Authentication
# ============================================================


def verify_admin_key(
    x_admin_key: str | None = Header(default=None),
):
    if x_admin_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing admin key",
        )

    if not secrets.compare_digest(
        x_admin_key,
        ADMIN_MASTER_KEY,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin key",
        )

    return True


# ============================================================
# Helpers
# ============================================================


def validate_limit_value(
    value: LimitValue,
    field_name: str,
) -> None:
    if value == "unlimited":
        return

    if value < 1:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be at least 1",
        )


def convert_limit(
    value: LimitValue,
) -> int | None:
    if value == "unlimited":
        return None

    return value


def limit_for_response(
    value: int | None,
) -> int | str:
    if value is None:
        return "unlimited"

    return value


def validate_model_config(
    models: dict[str, ModelLimitRequest],
) -> None:

    for model, config in models.items():
        if model not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {model}",
            )

        validate_limit_value(
            config.requests_per_minute,
            "requests_per_minute",
        )

        validate_limit_value(
            config.daily_token_limit,
            "daily_token_limit",
        )

        validate_limit_value(
            config.monthly_token_limit,
            "monthly_token_limit",
        )


def get_model_access(
    db: Session,
    api_key: APIKey,
    model: str,
) -> APIKeyModelAccess:

    access = (
        db.query(APIKeyModelAccess)
        .filter(
            APIKeyModelAccess.api_key_id == api_key.id,
            APIKeyModelAccess.model == model,
        )
        .first()
    )

    if access is None:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": (f"Model {model} is not enabled for this API key"),
                    "type": "model_access_error",
                    "code": "model_not_enabled",
                }
            },
        )

    if not access.enabled:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": (f"Model {model} is not enabled for this API key"),
                    "type": "model_access_error",
                    "code": "model_not_enabled",
                }
            },
        )

    return access


def reset_expired_quotas(
    access: APIKeyModelAccess,
) -> None:

    now = datetime.utcnow()

    # Daily reset
    if now.date() != access.daily_reset_at.date():
        access.daily_tokens_used = 0
        access.daily_reset_at = now

    # Monthly reset
    if (
        now.year != access.monthly_reset_at.year
        or now.month != access.monthly_reset_at.month
    ):
        access.monthly_tokens_used = 0
        access.monthly_reset_at = now


def check_rate_limit(
    access: APIKeyModelAccess,
) -> None:

    if access.requests_per_minute is None:
        return

    now = time.monotonic()

    key = (
        access.api_key_id,
        access.model,
    )

    requests = _rate_limit_store[key]

    window_start = now - 60

    while requests and requests[0] <= window_start:
        requests.popleft()

    if len(requests) >= access.requests_per_minute:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": ("Requests per minute limit exceeded"),
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
            headers={
                "Retry-After": "60",
            },
        )

    requests.append(now)


def estimate_tokens_from_chars(
    chars: int,
) -> int:

    return max(
        1,
        (chars + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE,
    )


def check_token_quota(
    access: APIKeyModelAccess,
    estimated_tokens: int,
) -> None:

    reset_expired_quotas(access)

    if access.daily_token_limit is not None and (
        access.daily_tokens_used + estimated_tokens > access.daily_token_limit
    ):
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": ("Daily token limit exceeded"),
                    "type": "quota_error",
                    "code": "daily_token_limit_exceeded",
                }
            },
        )

    if access.monthly_token_limit is not None and (
        access.monthly_tokens_used + estimated_tokens > access.monthly_token_limit
    ):
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": ("Monthly token limit exceeded"),
                    "type": "quota_error",
                    "code": "monthly_token_limit_exceeded",
                }
            },
        )


def record_token_usage(
    db: Session,
    access: APIKeyModelAccess,
    model: str,
    endpoint: str,
    prompt_tokens: int,
    completion_tokens: int,
    response_time_ms: float | None,
) -> None:

    total_tokens = prompt_tokens + completion_tokens

    reset_expired_quotas(access)

    access.daily_tokens_used += total_tokens
    access.monthly_tokens_used += total_tokens

    usage = APIUsage(
        api_key_id=access.api_key_id,
        model=model,
        endpoint=endpoint,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        response_time_ms=response_time_ms,
    )

    db.add(usage)
    db.commit()


def get_access_response(
    access: APIKeyModelAccess,
) -> dict:

    return {
        "model": access.model,
        "enabled": access.enabled,
        "requests_per_minute": limit_for_response(access.requests_per_minute),
        "daily_token_limit": limit_for_response(access.daily_token_limit),
        "daily_tokens_used": (access.daily_tokens_used),
        "daily_tokens_remaining": (
            "unlimited"
            if access.daily_token_limit is None
            else max(
                0,
                access.daily_token_limit - access.daily_tokens_used,
            )
        ),
        "monthly_token_limit": limit_for_response(access.monthly_token_limit),
        "monthly_tokens_used": (access.monthly_tokens_used),
        "monthly_tokens_remaining": (
            "unlimited"
            if access.monthly_token_limit is None
            else max(
                0,
                access.monthly_token_limit - access.monthly_tokens_used,
            )
        ),
    }


# ============================================================
# Health
# ============================================================


@app.get("/health")
async def health():

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "version": "3.0.0",
    }


# ============================================================
# Create API Key
# ============================================================


@app.post("/v1/keys")
async def create_api_key(
    request: CreateKeyRequest,
    _: bool = Depends(verify_admin_key),
    db: Session = Depends(get_db),
):

    validate_model_config(request.models)

    new_key = generate_api_key()

    # memory_limit is only meaningful when memory is enabled.
    memory_limit = request.memory_limit if request.memory_enabled else 0

    db_key = APIKey(
        name=request.name,
        key_prefix=get_key_prefix(new_key),
        key_hash=hash_api_key(new_key),
        active=True,
        memory_enabled=request.memory_enabled,
        memory_limit=memory_limit,
    )

    db.add(db_key)
    db.flush()

    # If models are omitted, enable both models
    # with default limits.
    model_configs = request.models

    if not model_configs:
        model_configs = {
            MODEL_NAME: ModelLimitRequest(),
            EMBEDDING_MODEL_NAME: ModelLimitRequest(),
        }

    for model, config in model_configs.items():
        access = APIKeyModelAccess(
            api_key_id=db_key.id,
            model=model,
            enabled=config.enabled,
            requests_per_minute=convert_limit(config.requests_per_minute),
            daily_token_limit=convert_limit(config.daily_token_limit),
            monthly_token_limit=convert_limit(config.monthly_token_limit),
            daily_tokens_used=0,
            monthly_tokens_used=0,
            daily_reset_at=datetime.utcnow(),
            monthly_reset_at=datetime.utcnow(),
        )

        db.add(access)

    db.commit()
    db.refresh(db_key)

    return {
        "id": db_key.id,
        "name": db_key.name,
        "api_key": new_key,
        "active": db_key.active,
        "memory_enabled": db_key.memory_enabled,
        "memory_limit": db_key.memory_limit,
        "models": [get_access_response(access) for access in db_key.model_access],
        "created_at": db_key.created_at,
        "warning": ("Save this API key now. It will not be shown again."),
    }


# ============================================================
# List API Keys
# ============================================================


@app.get("/v1/keys")
async def list_api_keys(
    _: bool = Depends(verify_admin_key),
    db: Session = Depends(get_db),
):

    keys = db.query(APIKey).order_by(APIKey.id.desc()).all()

    result = []

    for key in keys:
        for access in key.model_access:
            reset_expired_quotas(access)

        db.commit()

        result.append(
            {
                "id": key.id,
                "name": key.name,
                "prefix": key.key_prefix,
                "active": key.active,
                "memory_enabled": key.memory_enabled,
                "memory_limit": key.memory_limit,
                "created_at": key.created_at,
                "last_used_at": key.last_used_at,
                "models": [get_access_response(access) for access in key.model_access],
            }
        )

    return {
        "data": result,
    }


# ============================================================
# Change Model Access / Quota
# ============================================================


@app.patch("/v1/keys/{key_id}/quota")
async def update_key_quota(
    key_id: int,
    request: QuotaUpdateRequest,
    _: bool = Depends(verify_admin_key),
    db: Session = Depends(get_db),
):

    validate_model_config(request.models)

    key = db.query(APIKey).filter(APIKey.id == key_id).first()

    if key is None:
        raise HTTPException(
            status_code=404,
            detail="API key not found",
        )

    for model, config in request.models.items():
        access = (
            db.query(APIKeyModelAccess)
            .filter(
                APIKeyModelAccess.api_key_id == key.id,
                APIKeyModelAccess.model == model,
            )
            .first()
        )

        if access is None:
            access = APIKeyModelAccess(
                api_key_id=key.id,
                model=model,
                daily_tokens_used=0,
                monthly_tokens_used=0,
                daily_reset_at=datetime.utcnow(),
                monthly_reset_at=datetime.utcnow(),
            )

            db.add(access)

        access.enabled = config.enabled

        access.requests_per_minute = convert_limit(config.requests_per_minute)

        access.daily_token_limit = convert_limit(config.daily_token_limit)

        access.monthly_token_limit = convert_limit(config.monthly_token_limit)

    db.commit()

    accesses = (
        db.query(APIKeyModelAccess).filter(APIKeyModelAccess.api_key_id == key.id).all()
    )

    return {
        "id": key.id,
        "name": key.name,
        "models": [get_access_response(access) for access in accesses],
        "message": "Model access and quotas updated",
    }


# ============================================================
# Change Memory Settings
# ============================================================


@app.patch("/v1/keys/{key_id}/memory")
async def update_key_memory_settings(
    key_id: int,
    request: MemorySettingsUpdateRequest,
    _: bool = Depends(verify_admin_key),
    db: Session = Depends(get_db),
):

    key = db.query(APIKey).filter(APIKey.id == key_id).first()

    if key is None:
        raise HTTPException(
            status_code=404,
            detail="API key not found",
        )

    if request.memory_enabled is not None:
        key.memory_enabled = request.memory_enabled

    if request.memory_limit is not None:
        key.memory_limit = request.memory_limit

    # memory_limit is meaningless (and forced to 0) whenever memory
    # is disabled for this key.
    if not key.memory_enabled:
        key.memory_limit = 0

    db.commit()
    db.refresh(key)

    return {
        "id": key.id,
        "memory_enabled": key.memory_enabled,
        "memory_limit": key.memory_limit,
        "message": "Memory settings updated",
    }


# ============================================================
# Reset Usage
# ============================================================


@app.post("/v1/keys/{key_id}/reset-usage")
async def reset_key_usage(
    key_id: int,
    _: bool = Depends(verify_admin_key),
    db: Session = Depends(get_db),
):

    key = db.query(APIKey).filter(APIKey.id == key_id).first()

    if key is None:
        raise HTTPException(
            status_code=404,
            detail="API key not found",
        )

    accesses = (
        db.query(APIKeyModelAccess).filter(APIKeyModelAccess.api_key_id == key.id).all()
    )

    now = datetime.utcnow()

    for access in accesses:
        access.daily_tokens_used = 0
        access.monthly_tokens_used = 0
        access.daily_reset_at = now
        access.monthly_reset_at = now

    db.commit()

    return {
        "id": key.id,
        "message": "Usage counters reset",
        "models": [get_access_response(access) for access in accesses],
    }


# ============================================================
# Revoke API Key
# ============================================================


@app.delete("/v1/keys/{key_id}")
async def delete_api_key(
    key_id: int,
    _: bool = Depends(verify_admin_key),
    db: Session = Depends(get_db),
):
    key = db.query(APIKey).filter(APIKey.id == key_id).first()

    if key is None:
        raise HTTPException(
            status_code=404,
            detail="API key not found",
        )

    # Remove in-memory rate-limit state.
    for model in SUPPORTED_MODELS:
        _rate_limit_store.pop(
            (key.id, model),
            None,
        )

    # Hard delete.
    # PostgreSQL CASCADE will also permanently delete:
    # - api_key_model_access
    # - api_usage
    # - conversations
    # - user_memories
    db.delete(key)
    db.commit()

    return {
        "id": key_id,
        "deleted": True,
        "message": "API key and all associated data permanently deleted",
    }


# ============================================================
# Key Usage
# ============================================================


@app.get("/v1/keys/{key_id}/usage")
async def get_key_usage(
    key_id: int,
    _: bool = Depends(verify_admin_key),
    db: Session = Depends(get_db),
):

    key = db.query(APIKey).filter(APIKey.id == key_id).first()

    if key is None:
        raise HTTPException(
            status_code=404,
            detail="API key not found",
        )

    accesses = (
        db.query(APIKeyModelAccess).filter(APIKeyModelAccess.api_key_id == key.id).all()
    )

    for access in accesses:
        reset_expired_quotas(access)

    db.commit()

    usage = (
        db.query(APIUsage)
        .filter(APIUsage.api_key_id == key.id)
        .order_by(APIUsage.created_at.desc())
        .all()
    )

    return {
        "api_key_id": key.id,
        "name": key.name,
        "models": [get_access_response(access) for access in accesses],
        "usage": [
            {
                "id": item.id,
                "model": item.model,
                "endpoint": item.endpoint,
                "prompt_tokens": item.prompt_tokens,
                "completion_tokens": item.completion_tokens,
                "total_tokens": item.total_tokens,
                "response_time_ms": item.response_time_ms,
                "created_at": item.created_at,
            }
            for item in usage
        ],
    }


# ============================================================
# All Usage
# ============================================================


@app.get("/v1/usage")
async def get_all_usage(
    _: bool = Depends(verify_admin_key),
    db: Session = Depends(get_db),
):

    usage = (
        db.query(APIUsage, APIKey)
        .join(
            APIKey,
            APIUsage.api_key_id == APIKey.id,
        )
        .order_by(APIUsage.created_at.desc())
        .all()
    )

    return {
        "data": [
            {
                "id": item.id,
                "api_key_id": key.id,
                "api_key_name": key.name,
                "model": item.model,
                "endpoint": item.endpoint,
                "prompt_tokens": item.prompt_tokens,
                "completion_tokens": item.completion_tokens,
                "total_tokens": item.total_tokens,
                "response_time_ms": item.response_time_ms,
                "created_at": item.created_at,
            }
            for item, key in usage
        ]
    }


# ============================================================
# Models
# ============================================================


@app.get("/v1/models")
async def list_models(
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    accesses = (
        db.query(APIKeyModelAccess)
        .filter(
            APIKeyModelAccess.api_key_id == api_key.id,
            APIKeyModelAccess.enabled.is_(True),
        )
        .all()
    )

    return {
        "object": "list",
        "data": [
            {
                "id": access.model,
                "object": "model",
                "owned_by": "local",
            }
            for access in accesses
        ],
    }


# ============================================================
# Validate Chat Request
# ============================================================


def validate_chat_request(
    request: ChatCompletionRequest,
) -> None:

    if request.model != MODEL_NAME:
        raise HTTPException(
            status_code=400,
            detail=f"Model must be {MODEL_NAME}",
        )

    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Messages cannot be empty",
        )

    if len(request.messages) > MAX_MESSAGES:
        raise HTTPException(
            status_code=400,
            detail=(f"Maximum {MAX_MESSAGES} messages are allowed per request"),
        )

    total_chars = 0

    allowed_roles = {
        "system",
        "user",
        "assistant",
        "tool",
    }

    for index, message in enumerate(request.messages):
        if message.role not in allowed_roles:
            raise HTTPException(
                status_code=400,
                detail=(f"Invalid role at message index {index}"),
            )

        if not message.content:
            raise HTTPException(
                status_code=400,
                detail=(f"Message at index {index} cannot be empty"),
            )

        if len(message.content) > MAX_MESSAGE_CHARS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Message at index {index} exceeds "
                    f"maximum {MAX_MESSAGE_CHARS} "
                    "characters"
                ),
            )

        total_chars += len(message.content)

    if total_chars > MAX_TOTAL_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(f"Total input exceeds maximum {MAX_TOTAL_INPUT_CHARS} characters"),
        )


# ============================================================
# Memory Helpers
# ============================================================


def get_last_user_message(
    messages: list[Message],
) -> str | None:

    for message in reversed(messages):
        if message.role == "user":
            return message.content

    return None


async def build_memory_context(
    db: Session,
    api_key: APIKey,
    last_user_message: str | None,
) -> str | None:
    """
    Best-effort retrieval of relevant long-term memories for this
    key. Never raises - a temporary embedding/DB problem should not
    break a normal chat response.
    """

    if not last_user_message:
        return None

    try:
        query_embedding = await generate_embedding(last_user_message)

        relevant_memories = load_relevant_memories(
            db,
            api_key.id,
            query_embedding,
        )

        return build_memory_context_message(relevant_memories)

    except Exception:
        logger.warning(
            "Memory retrieval failed for api_key_id=%s",
            api_key.id,
            exc_info=True,
        )

        return None


# ============================================================
# Build Ollama Chat Payload
# ============================================================


def build_chat_payload(
    request: ChatCompletionRequest,
    memory_context: str | None = None,
) -> dict:
    messages = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in request.messages
    ]

    completion_instruction = (
        f"Answer the user's request completely and naturally within "
        f"{request.max_tokens} tokens. Prioritize finishing the answer "
        "over adding extra detail. Keep the response concise enough to "
        "reach a complete ending. Do not begin a new section, list item, "
        "or sentence unless you have enough space to finish it. "
        "Always aim to end at a natural sentence boundary."
    )

    system_additions = completion_instruction

    if memory_context:
        system_additions = memory_context + "\n\n" + completion_instruction

    if messages and messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system_additions
    else:
        messages.insert(
            0,
            {
                "role": "system",
                "content": system_additions,
            },
        )

    return {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "num_ctx": 2048,
            "num_predict": request.max_tokens,
            "temperature": request.temperature,
        },
    }


# ============================================================
# Chat Completions
# ============================================================


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    validate_chat_request(request)

    access = get_model_access(
        db,
        api_key,
        MODEL_NAME,
    )

    check_rate_limit(access)

    input_chars = sum(len(message.content) for message in request.messages)

    estimated_prompt_tokens = estimate_tokens_from_chars(input_chars)

    estimated_tokens = estimated_prompt_tokens + request.max_tokens

    check_token_quota(
        access,
        estimated_tokens,
    )

    db.commit()

    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    # ========================================================
    # Memory (MODE 2 only) - retrieval happens before the model
    # call; creation/update happens after, without blocking the
    # response. Mode 1 keys (memory_enabled=False) never touch
    # this at all.
    # ========================================================

    last_user_message: str | None = None
    memory_context: str | None = None

    if api_key.memory_enabled:
        last_user_message = get_last_user_message(request.messages)

        memory_context = await build_memory_context(
            db,
            api_key,
            last_user_message,
        )

    # ========================================================
    # Streaming
    # ========================================================

    if request.stream:
        return StreamingResponse(
            stream_chat_response(
                request=request,
                request_id=request_id,
                api_key_id=api_key.id,
                memory_context=memory_context,
                memory_enabled=api_key.memory_enabled,
                last_user_message=last_user_message,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )

    # ========================================================
    # Non-streaming
    # ========================================================

    payload = build_chat_payload(request, memory_context)

    try:
        start_time = time.perf_counter()

        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
            )

            response.raise_for_status()

        response_time_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail={
                "error": {
                    "message": ("Ollama request timed out"),
                    "type": "upstream_timeout",
                    "code": "ollama_timeout",
                }
            },
        )

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": ("Ollama returned an upstream error"),
                    "type": "upstream_error",
                    "code": "ollama_error",
                    "status": (exc.response.status_code),
                }
            },
        )

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": (f"Ollama request failed: {str(exc)}"),
                    "type": "upstream_error",
                    "code": ("ollama_connection_error"),
                }
            },
        )

    try:
        data = response.json()

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": ("Invalid response received from Ollama"),
                    "type": "upstream_error",
                    "code": ("invalid_ollama_response"),
                }
            },
        )

    message = data.get("message") or {}

    content = message.get("content") or ""

    if not content:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": ("Ollama returned an empty response"),
                    "type": "upstream_error",
                    "code": "empty_response",
                }
            },
        )

    done_reason = data.get(
        "done_reason",
        "stop",
    )

    finish_reason = "length" if done_reason == "length" else "stop"

    prompt_tokens = (
        data.get(
            "prompt_eval_count",
            0,
        )
        or 0
    )

    completion_tokens = (
        data.get(
            "eval_count",
            0,
        )
        or 0
    )

    total_tokens = prompt_tokens + completion_tokens

    # Record actual Ollama usage.
    access = get_model_access(
        db,
        api_key,
        MODEL_NAME,
    )

    record_token_usage(
        db=db,
        access=access,
        model=MODEL_NAME,
        endpoint="/v1/chat/completions",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_time_ms=response_time_ms,
    )

    if api_key.memory_enabled and last_user_message:
        background_tasks.add_task(
            remember_exchange_background,
            api_key.id,
            last_user_message,
            content,
        )

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "response_time_ms": response_time_ms,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


# ============================================================
# Streaming Generator
# ============================================================


async def stream_chat_response(
    request: ChatCompletionRequest,
    request_id: str,
    api_key_id: int,
    memory_context: str | None = None,
    memory_enabled: bool = False,
    last_user_message: str | None = None,
) -> AsyncGenerator[str, None]:

    payload = build_chat_payload(request, memory_context)
    payload["stream"] = True

    prompt_tokens = 0
    completion_tokens = 0
    full_content_parts: list[str] = []

    start_time = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                    except json.JSONDecodeError:
                        continue

                    prompt_tokens = (
                        data.get(
                            "prompt_eval_count",
                            prompt_tokens,
                        )
                        or prompt_tokens
                    )

                    completion_tokens = (
                        data.get(
                            "eval_count",
                            completion_tokens,
                        )
                        or completion_tokens
                    )

                    message = data.get("message") or {}

                    content = message.get(
                        "content",
                        "",
                    )

                    if content:
                        full_content_parts.append(content)

                        chunk = {
                            "id": request_id,
                            "object": ("chat.completion.chunk"),
                            "created": int(time.time()),
                            "model": MODEL_NAME,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None,
                                }
                            ],
                        }

                        yield ("data: " + json.dumps(chunk) + "\n\n")

                    if data.get("done"):
                        finish_reason = (
                            "length" if data.get("done_reason") == "length" else "stop"
                        )

                        final_chunk = {
                            "id": request_id,
                            "object": ("chat.completion.chunk"),
                            "created": int(time.time()),
                            "model": MODEL_NAME,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": (finish_reason),
                                }
                            ],
                        }

                        yield ("data: " + json.dumps(final_chunk) + "\n\n")

                        # Record streaming usage.
                        response_time_ms = round(
                            (time.perf_counter() - start_time) * 1000,
                            2,
                        )

                        # Create a short-lived DB session.
                        from .database import SessionLocal

                        db = SessionLocal()

                        try:
                            access = (
                                db.query(APIKeyModelAccess)
                                .filter(
                                    APIKeyModelAccess.api_key_id == api_key_id,
                                    APIKeyModelAccess.model == MODEL_NAME,
                                )
                                .first()
                            )

                            if access is not None:
                                record_token_usage(
                                    db=db,
                                    access=access,
                                    model=MODEL_NAME,
                                    endpoint=("/v1/chat/completions"),
                                    prompt_tokens=(prompt_tokens),
                                    completion_tokens=(completion_tokens),
                                    response_time_ms=(response_time_ms),
                                )

                            if memory_enabled and last_user_message:
                                full_content = "".join(full_content_parts)

                                if full_content:
                                    try:
                                        stream_api_key = (
                                            db.query(APIKey)
                                            .filter(APIKey.id == api_key_id)
                                            .first()
                                        )

                                        if (
                                            stream_api_key is not None
                                            and stream_api_key.memory_enabled
                                        ):
                                            await remember_exchange(
                                                db,
                                                stream_api_key,
                                                last_user_message,
                                                full_content,
                                            )

                                    except Exception:
                                        logger.warning(
                                            "Memory extraction failed for "
                                            "api_key_id=%s",
                                            api_key_id,
                                            exc_info=True,
                                        )

                        finally:
                            db.close()

                        yield "data: [DONE]\n\n"

    except Exception as exc:
        error_chunk = {
            "error": {
                "message": (f"Streaming failed: {str(exc)}"),
                "type": "upstream_error",
                "code": "ollama_stream_error",
            }
        }

        yield ("data: " + json.dumps(error_chunk) + "\n\n")


# ============================================================
# Embeddings
# ============================================================


@app.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):

    if request.model != EMBEDDING_MODEL_NAME:
        raise HTTPException(
            status_code=400,
            detail=(f"Model must be {EMBEDDING_MODEL_NAME}"),
        )

    access = get_model_access(
        db,
        api_key,
        EMBEDDING_MODEL_NAME,
    )

    check_rate_limit(access)

    if isinstance(request.input, str):
        inputs = [request.input]

    else:
        inputs = request.input

    if not inputs:
        raise HTTPException(
            status_code=400,
            detail="Input cannot be empty",
        )

    if len(inputs) > MAX_EMBEDDING_BATCH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Maximum {MAX_EMBEDDING_BATCH} texts are allowed per embedding request"
            ),
        )

    total_chars = 0

    for index, text in enumerate(inputs):
        if not isinstance(text, str):
            raise HTTPException(
                status_code=400,
                detail=(f"Input at index {index} must be a string"),
            )

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=(f"Input at index {index} cannot be empty"),
            )

        if len(text) > MAX_EMBEDDING_CHARS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Input at index {index} "
                    f"exceeds maximum "
                    f"{MAX_EMBEDDING_CHARS} "
                    "characters"
                ),
            )

        total_chars += len(text)

    if total_chars > MAX_TOTAL_EMBEDDING_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Total embedding input exceeds "
                f"maximum "
                f"{MAX_TOTAL_EMBEDDING_CHARS} "
                "characters"
            ),
        )

    estimated_tokens = estimate_tokens_from_chars(total_chars)

    check_token_quota(
        access,
        estimated_tokens,
    )

    db.commit()

    embeddings = []

    start_time = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            for text in inputs:
                response = await client.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={
                        "model": (EMBEDDING_MODEL_NAME),
                        "input": text,
                        "keep_alive": (OLLAMA_KEEP_ALIVE),
                    },
                )

                response.raise_for_status()

                data = response.json()

                vector_list = data.get("embeddings")

                if not vector_list:
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": {
                                "message": ("Ollama returned an empty embedding"),
                                "type": ("upstream_error"),
                                "code": ("empty_embedding"),
                            }
                        },
                    )

                embedding = vector_list[0]

                if len(embedding) != EMBEDDING_DIMENSIONS:
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": {
                                "message": ("Unexpected embedding dimensions"),
                                "type": ("upstream_error"),
                                "code": ("invalid_embedding_dimensions"),
                                "expected": (EMBEDDING_DIMENSIONS),
                                "actual": len(embedding),
                            }
                        },
                    )

                embeddings.append(embedding)

    except HTTPException:
        raise

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail={
                "error": {
                    "message": ("Ollama embedding request timed out"),
                    "type": "upstream_timeout",
                    "code": "ollama_timeout",
                }
            },
        )

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": ("Ollama returned an embedding error"),
                    "type": "upstream_error",
                    "code": ("ollama_embedding_error"),
                    "status": (exc.response.status_code),
                }
            },
        )

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": (f"Ollama embedding request failed: {str(exc)}"),
                    "type": "upstream_error",
                    "code": ("ollama_connection_error"),
                }
            },
        )

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": ("Invalid JSON response from Ollama"),
                    "type": "upstream_error",
                    "code": ("invalid_ollama_response"),
                }
            },
        )

    response_time_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    # Ollama embed currently does not expose
    # token counts in this API response.
    # Therefore this is an estimate.
    embedding_tokens = estimated_tokens

    record_token_usage(
        db=db,
        access=access,
        model=EMBEDDING_MODEL_NAME,
        endpoint="/v1/embeddings",
        prompt_tokens=embedding_tokens,
        completion_tokens=0,
        response_time_ms=response_time_ms,
    )

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": embedding,
            }
            for index, embedding in enumerate(embeddings)
        ],
        "model": EMBEDDING_MODEL_NAME,
        "response_time_ms": response_time_ms,
        "usage": {
            "prompt_tokens": embedding_tokens,
            "total_tokens": embedding_tokens,
        },
    }


# ============================================================
# Root
# ============================================================


@app.get("/")
async def root():

    return {
        "name": "Qwen API",
        "version": "3.0.0",
        "status": "running",
        "chat_model": MODEL_NAME,
        "embedding_model": (EMBEDDING_MODEL_NAME),
        "docs": "/docs",
    }
