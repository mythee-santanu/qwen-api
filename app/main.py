import os
import secrets

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .dependencies import get_current_api_key
from .models import APIKey
from .security import generate_api_key, get_key_prefix, hash_api_key


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Qwen API",
    version="1.0.0",
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://host.docker.internal:11434",
)

MODEL_NAME = "qwen3:1.7b"

ADMIN_MASTER_KEY = os.getenv("ADMIN_MASTER_KEY")

if not ADMIN_MASTER_KEY:
    raise RuntimeError("ADMIN_MASTER_KEY is not configured")


class CreateKeyRequest(BaseModel):
    name: str


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False


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


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


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
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: APIKey = Depends(get_current_api_key),
):
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming will be added in the next step.",
        )

    if request.model != MODEL_NAME:
        raise HTTPException(
            status_code=400,
            detail=f"Model must be {MODEL_NAME}",
        )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ],
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "num_ctx": 2048,
            "temperature": 0.7,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
            )

            response.raise_for_status()

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {str(exc)}",
        )

    data = response.json()

    content = data.get("message", {}).get("content", "")

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": MODEL_NAME,
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
