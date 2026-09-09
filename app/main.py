import os
import secrets
import time

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .dependencies import get_current_api_key
from .models import APIKey
from .security import generate_api_key, get_key_prefix, hash_api_key


# --------------------------------------------------
# Database
# --------------------------------------------------

Base.metadata.create_all(bind=engine)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Qwen API",
    version="1.1.0",
)


# --------------------------------------------------
# Ollama Configuration
# --------------------------------------------------

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://host.docker.internal:11434",
)

# Chat / generation model
MODEL_NAME = "qwen3.5:0.8b"

# Embedding model
EMBEDDING_MODEL_NAME = "qwen3-embedding:0.6b"

ADMIN_MASTER_KEY = os.getenv("ADMIN_MASTER_KEY")

if not ADMIN_MASTER_KEY:
    raise RuntimeError("ADMIN_MASTER_KEY is not configured")


# --------------------------------------------------
# Request Models
# --------------------------------------------------


class CreateKeyRequest(BaseModel):
    name: str


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]


# --------------------------------------------------
# Admin Authentication
# --------------------------------------------------


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


# --------------------------------------------------
# Health
# --------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }


# --------------------------------------------------
# Create API Key
# --------------------------------------------------


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
        "warning": "Save this API key now. It will not be shown again.",
    }


# --------------------------------------------------
# List API Keys
# --------------------------------------------------


@app.get("/v1/keys")
async def list_api_keys(
    _: bool = Depends(verify_admin_key),
    db: Session = Depends(get_db),
):
    keys = db.query(APIKey).order_by(APIKey.id.desc()).all()

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


# --------------------------------------------------
# Revoke API Key
# --------------------------------------------------


@app.delete("/v1/keys/{key_id}")
async def revoke_api_key(
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

    key.active = False
    db.commit()

    return {
        "id": key.id,
        "active": False,
        "message": "API key revoked",
    }


# --------------------------------------------------
# Models
# --------------------------------------------------


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


# --------------------------------------------------
# Chat Completions
# --------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: APIKey = Depends(get_current_api_key),
):
    # ----------------------------------------------
    # Streaming
    # ----------------------------------------------

    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming will be added in the next step.",
        )

    # ----------------------------------------------
    # Model validation
    # ----------------------------------------------

    if request.model != MODEL_NAME:
        raise HTTPException(
            status_code=400,
            detail=f"Model must be {MODEL_NAME}",
        )

    # ----------------------------------------------
    # Ollama payload
    # ----------------------------------------------

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer directly and concisely. "
                    "Use 1-2 complete sentences. "
                    "Do not add unnecessary explanation."
                ),
            },
            *[
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
        ],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "num_ctx": 2048,
            "num_predict": 96,
            "temperature": 0.2,
        },
    }

    # ----------------------------------------------
    # Call Ollama + measure response time
    # ----------------------------------------------

    try:
        start_time = time.perf_counter()

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
            )

            response.raise_for_status()

        response_time_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {str(exc)}",
        )

    # ----------------------------------------------
    # Parse Ollama response
    # ----------------------------------------------

    try:
        data = response.json()

        content = data.get(
            "message",
            {},
        ).get(
            "content",
            "",
        )

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Invalid response received from Ollama",
        )

    if not content:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an empty response",
        )

    # ----------------------------------------------
    # OpenAI-compatible response
    # ----------------------------------------------

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": MODEL_NAME,
        "response_time_ms": response_time_ms,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
    }


# --------------------------------------------------
# Embeddings
# --------------------------------------------------


@app.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    api_key: APIKey = Depends(get_current_api_key),
):
    # ----------------------------------------------
    # Model validation
    # ----------------------------------------------

    if request.model != EMBEDDING_MODEL_NAME:
        raise HTTPException(
            status_code=400,
            detail=f"Model must be {EMBEDDING_MODEL_NAME}",
        )

    # ----------------------------------------------
    # Validate input
    # ----------------------------------------------

    if isinstance(request.input, str):
        inputs = [request.input]
    else:
        inputs = request.input

    if not inputs:
        raise HTTPException(
            status_code=400,
            detail="Input cannot be empty",
        )

    # ----------------------------------------------
    # Generate embeddings
    # ----------------------------------------------

    embeddings = []

    try:
        start_time = time.perf_counter()

        async with httpx.AsyncClient(timeout=300.0) as client:
            for text in inputs:
                response = await client.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={
                        "model": EMBEDDING_MODEL_NAME,
                        "input": text,
                    },
                )

                response.raise_for_status()

                data = response.json()

                vector_list = data.get("embeddings")

                if not vector_list:
                    raise HTTPException(
                        status_code=502,
                        detail="Ollama returned an empty embedding",
                    )

                embeddings.append(vector_list[0])

        response_time_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

    except HTTPException:
        raise

    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama embedding request failed: {str(exc)}",
        )

    # ----------------------------------------------
    # OpenAI-compatible embedding response
    # ----------------------------------------------

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
