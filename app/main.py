import json
import os
import secrets
import time
import uuid
from collections import defaultdict, deque
from typing import AsyncGenerator

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .dependencies import get_current_api_key
from .models import APIKey
from .security import generate_api_key, get_key_prefix, hash_api_key


# ============================================================
# Database
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Qwen API",
    version="2.0.0",
    description="OpenAI-compatible local Qwen API",
)


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

# Rate limit
# Designed for a 1 CPU VPS.
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW_SECONDS = 60

# Ollama
OLLAMA_TIMEOUT = 300.0
OLLAMA_KEEP_ALIVE = "30m"

ADMIN_MASTER_KEY = os.getenv("ADMIN_MASTER_KEY")

if not ADMIN_MASTER_KEY:
    raise RuntimeError("ADMIN_MASTER_KEY is not configured")


# ============================================================
# Simple in-memory rate limiter
# ============================================================

_rate_limit_store: dict[int, deque[float]] = defaultdict(deque)


def check_rate_limit(api_key: APIKey) -> None:
    """
    Simple per-API-key sliding-window rate limiter.

    This is intentionally in-memory because this deployment
    uses one FastAPI worker on a small VPS.
    """

    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    requests = _rate_limit_store[api_key.id]

    while requests and requests[0] <= window_start:
        requests.popleft()

    if len(requests) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": "Rate limit exceeded. Please try again later.",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
            headers={
                "Retry-After": str(RATE_LIMIT_WINDOW_SECONDS),
            },
        )

    requests.append(now)


# ============================================================
# Request Models
# ============================================================

class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


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
        default=96,
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
# Health
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "limits": {
            "max_messages": MAX_MESSAGES,
            "max_message_chars": MAX_MESSAGE_CHARS,
            "max_total_input_chars": MAX_TOTAL_INPUT_CHARS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "max_embedding_batch": MAX_EMBEDDING_BATCH,
            "max_embedding_chars": MAX_EMBEDDING_CHARS,
            "rate_limit_requests": RATE_LIMIT_REQUESTS,
            "rate_limit_window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        },
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
    new_key = generate_api_key()

    db_key = APIKey(
        name=request.name,
        key_prefix=get_key_prefix(new_key),
        key_hash=hash_api_key(new_key),
        active=True,
    )

    db.add(db_key)
    db.commit()
    db.refresh(db_key)

    return {
        "id": db_key.id,
        "name": db_key.name,
        "api_key": new_key,
        "created_at": db_key.created_at,
        "warning": (
            "Save this API key now. "
            "It will not be shown again."
        ),
    }


# ============================================================
# List API Keys
# ============================================================

@app.get("/v1/keys")
async def list_api_keys(
    _: bool = Depends(verify_admin_key),
    db: Session = Depends(get_db),
):
    keys = (
        db.query(APIKey)
        .order_by(APIKey.id.desc())
        .all()
    )

    return {
        "data": [
            {
                "id": key.id,
                "name": key.name,
                "prefix": key.key_prefix,
                "active": key.active,
                "created_at": key.created_at,
                "last_used_at": key.last_used_at,
            }
            for key in keys
        ]
    }


# ============================================================
# Revoke API Key
# ============================================================

@app.delete("/v1/keys/{key_id}")
async def revoke_api_key(
    key_id: int,
    _: bool = Depends(verify_admin_key),
    db: Session = Depends(get_db),
):
    key = (
        db.query(APIKey)
        .filter(APIKey.id == key_id)
        .first()
    )

    if key is None:
        raise HTTPException(
            status_code=404,
            detail="API key not found",
        )

    key.active = False
    db.commit()

    # Remove rate-limit state for this key.
    _rate_limit_store.pop(key.id, None)

    return {
        "id": key.id,
        "active": False,
        "message": "API key revoked",
    }


# ============================================================
# Models
# ============================================================

@app.get("/v1/models")
async def list_models(
    api_key: APIKey = Depends(get_current_api_key),
):
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            },
            {
                "id": EMBEDDING_MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            },
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
            detail=(
                f"Maximum {MAX_MESSAGES} messages "
                "are allowed per request"
            ),
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
                detail=(
                    f"Invalid role at message index {index}"
                ),
            )

        if not message.content:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Message at index {index} "
                    "cannot be empty"
                ),
            )

        if len(message.content) > MAX_MESSAGE_CHARS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Message at index {index} exceeds "
                    f"maximum {MAX_MESSAGE_CHARS} characters"
                ),
            )

        total_chars += len(message.content)

    if total_chars > MAX_TOTAL_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Total input exceeds maximum "
                f"{MAX_TOTAL_INPUT_CHARS} characters"
            ),
        )


# ============================================================
# Build Ollama Chat Payload
# ============================================================

def build_chat_payload(
    request: ChatCompletionRequest,
) -> dict:

    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ],
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
    api_key: APIKey = Depends(get_current_api_key),
):

    check_rate_limit(api_key)
    validate_chat_request(request)

    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    # ========================================================
    # Streaming
    # ========================================================

    if request.stream:

        return StreamingResponse(
            stream_chat_response(
                request=request,
                request_id=request_id,
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

    payload = build_chat_payload(request)

    try:
        start_time = time.perf_counter()

        async with httpx.AsyncClient(
            timeout=OLLAMA_TIMEOUT
        ) as client:

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
                    "message": "Ollama request timed out",
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
                    "message": (
                        "Ollama returned an upstream error"
                    ),
                    "type": "upstream_error",
                    "code": "ollama_error",
                    "status": exc.response.status_code,
                }
            },
        )

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": (
                        f"Ollama request failed: {str(exc)}"
                    ),
                    "type": "upstream_error",
                    "code": "ollama_connection_error",
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
                    "message": (
                        "Invalid response received from Ollama"
                    ),
                    "type": "upstream_error",
                    "code": "invalid_ollama_response",
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
                    "message": (
                        "Ollama returned an empty response"
                    ),
                    "type": "upstream_error",
                    "code": "empty_response",
                }
            },
        )

    done_reason = data.get(
        "done_reason",
        "stop",
    )

    finish_reason = (
        "length"
        if done_reason == "length"
        else "stop"
    )

    prompt_tokens = data.get(
        "prompt_eval_count",
        0,
    ) or 0

    completion_tokens = data.get(
        "eval_count",
        0,
    ) or 0

    total_tokens = (
        prompt_tokens +
        completion_tokens
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
) -> AsyncGenerator[str, None]:

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ],
        "stream": True,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "num_ctx": 2048,
            "num_predict": request.max_tokens,
            "temperature": request.temperature,
        },
    }

    try:

        async with httpx.AsyncClient(
            timeout=OLLAMA_TIMEOUT
        ) as client:

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

                    message = data.get(
                        "message"
                    ) or {}

                    content = message.get(
                        "content",
                        "",
                    )

                    if content:

                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": MODEL_NAME,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": content,
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }

                        yield (
                            f"data: {json.dumps(chunk)}\n\n"
                        )

                    if data.get("done"):

                        finish_reason = (
                            "length"
                            if data.get("done_reason")
                            == "length"
                            else "stop"
                        )

                        final_chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": MODEL_NAME,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }

                        yield (
                            f"data: "
                            f"{json.dumps(final_chunk)}\n\n"
                        )

                        yield "data: [DONE]\n\n"

    except Exception as exc:

        error_chunk = {
            "error": {
                "message": f"Streaming failed: {str(exc)}",
                "type": "upstream_error",
                "code": "ollama_stream_error",
            }
        }

        yield (
            f"data: {json.dumps(error_chunk)}\n\n"
        )


# ============================================================
# Embeddings
# ============================================================

@app.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    api_key: APIKey = Depends(get_current_api_key),
):

    check_rate_limit(api_key)

    # ========================================================
    # Model validation
    # ========================================================

    if request.model != EMBEDDING_MODEL_NAME:
        raise HTTPException(
            status_code=400,
            detail=f"Model must be {EMBEDDING_MODEL_NAME}",
        )

    # ========================================================
    # Normalize input
    # ========================================================

    if isinstance(request.input, str):
        inputs = [request.input]
    else:
        inputs = request.input

    if not inputs:
        raise HTTPException(
            status_code=400,
            detail="Input cannot be empty",
        )

    # ========================================================
    # Batch limit
    # ========================================================

    if len(inputs) > MAX_EMBEDDING_BATCH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Maximum {MAX_EMBEDDING_BATCH} texts "
                "are allowed per embedding request"
            ),
        )

    # ========================================================
    # Text validation
    # ========================================================

    total_chars = 0

    for index, text in enumerate(inputs):

        if not isinstance(text, str):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Input at index {index} "
                    "must be a string"
                ),
            )

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Input at index {index} "
                    "cannot be empty"
                ),
            )

        if len(text) > MAX_EMBEDDING_CHARS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Input at index {index} exceeds "
                    f"maximum {MAX_EMBEDDING_CHARS} characters"
                ),
            )

        total_chars += len(text)

    if total_chars > MAX_TOTAL_EMBEDDING_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Total embedding input exceeds "
                f"maximum {MAX_TOTAL_EMBEDDING_CHARS} characters"
            ),
        )

    # ========================================================
    # Generate embeddings
    # ========================================================

    embeddings = []

    start_time = time.perf_counter()

    try:

        async with httpx.AsyncClient(
            timeout=OLLAMA_TIMEOUT
        ) as client:

            # Sequential intentionally.
            # VPS has only 1 CPU core.

            for text in inputs:

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

                vector_list = data.get(
                    "embeddings"
                )

                if not vector_list:
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": {
                                "message": (
                                    "Ollama returned "
                                    "an empty embedding"
                                ),
                                "type": "upstream_error",
                                "code": "empty_embedding",
                            }
                        },
                    )

                embedding = vector_list[0]

                if len(embedding) != EMBEDDING_DIMENSIONS:
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": {
                                "message": (
                                    "Unexpected embedding "
                                    "dimensions"
                                ),
                                "type": "upstream_error",
                                "code": "invalid_embedding_dimensions",
                                "expected": EMBEDDING_DIMENSIONS,
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
                    "message": (
                        "Ollama embedding request timed out"
                    ),
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
                    "message": (
                        "Ollama returned an embedding error"
                    ),
                    "type": "upstream_error",
                    "code": "ollama_embedding_error",
                    "status": exc.response.status_code,
                }
            },
        )

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": (
                        f"Ollama embedding request failed: "
                        f"{str(exc)}"
                    ),
                    "type": "upstream_error",
                    "code": "ollama_connection_error",
                }
            },
        )

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "message": (
                        "Invalid JSON response from Ollama"
                    ),
                    "type": "upstream_error",
                    "code": "invalid_ollama_response",
                }
            },
        )

    response_time_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    # ========================================================
    # OpenAI-compatible response
    # ========================================================

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
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }


# ============================================================
# Root
# ============================================================

@app.get("/")
async def root():
    return {
        "name": "Qwen API",
        "version": "2.0.0",
        "status": "running",
        "chat_model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "docs": "/docs",
    }