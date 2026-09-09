# 🤖 Qwen API

<p align="center">
  <strong>🚀 OpenAI-Compatible Local Qwen API</strong>
  <br>
  <em>FastAPI + Ollama + PostgreSQL + Docker</em>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)

</p>

<p align="center">

![Qwen](https://img.shields.io/badge/Qwen-Local_LLM-FF6A00?style=flat-square)
![API](https://img.shields.io/badge/API-OpenAI_Compatible-10B981?style=flat-square)
![Streaming](https://img.shields.io/badge/Streaming-SSE-8B5CF6?style=flat-square)
![Embeddings](https://img.shields.io/badge/Embeddings-1024D-F59E0B?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

</p>

---

## ✨ Overview

**Qwen API** is a lightweight, self-hosted, OpenAI-compatible API built with **FastAPI** and powered by **Ollama**.

It provides local access to Qwen models through a clean REST API with authentication, API-key management, rate limiting, streaming responses, and embeddings.

```text
                         🌐 Client
                            │
                            ▼
                  ┌────────────────────┐
                  │     🚀 Qwen API    │
                  │      FastAPI       │
                  └─────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          🐘 PostgreSQL             🦙 Ollama
          API Keys DB                  │
                                      │
                           ┌──────────┴──────────┐
                           ▼                     ▼
                    🧠 Qwen Chat          🔢 Embeddings
                    qwen3.5:0.8b         qwen3-embedding:0.6b
```

---

## 🌟 Features

| Feature | Status |
|---|:---:|
| 🤖 Qwen Chat Completions | ✅ |
| 🌊 Streaming / SSE | ✅ |
| 🔢 Embeddings | ✅ |
| 🔑 API Key Authentication | ✅ |
| 🛡️ Admin Key Management | ✅ |
| 🚦 Per-Key Rate Limiting | ✅ |
| 🐘 PostgreSQL Storage | ✅ |
| 🐳 Docker Deployment | ✅ |
| 📚 Swagger / OpenAPI | ✅ |
| ⚡ Async HTTP Requests | ✅ |
| 🔒 Hashed API Keys | ✅ |
| 📏 Request Size Limits | ✅ |
| 💚 Health Endpoint | ✅ |

---

# 🧠 Models

### 💬 Chat Model

```text
qwen3.5:0.8b
```

### 🔢 Embedding Model

```text
qwen3-embedding:0.6b
```

### 📐 Embedding Dimensions

```text
1024
```

---

# 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────┐
│                     🌍 Client                       │
└─────────────────────────┬───────────────────────────┘
                          │
                          │ HTTP / HTTPS
                          ▼
┌─────────────────────────────────────────────────────┐
│                  🚀 FastAPI                         │
│                    Qwen API                         │
│                                                     │
│  🔑 Authentication                                  │
│  🚦 Rate Limiting                                   │
│  💬 Chat Completions                                │
│  🌊 Streaming                                       │
│  🔢 Embeddings                                      │
└───────────────┬───────────────────┬─────────────────┘
                │                   │
                ▼                   ▼
      ┌─────────────────┐   ┌─────────────────────┐
      │ 🐘 PostgreSQL   │   │ 🦙 Ollama            │
      │                 │   │                     │
      │ API Keys        │   │ qwen3.5:0.8b        │
      │ Key Hashes      │   │ qwen3-embedding     │
      │ Usage Metadata  │   │                     │
      └─────────────────┘   └─────────────────────┘
```

---

# ⚡ Quick Start

## 1️⃣ Clone / enter project

```bash
cd ~/qwen-api
```

## 2️⃣ Configure environment

Create:

```bash
nano .env
```

Example:

```env
DATABASE_URL=postgresql+psycopg://qwen:YOUR_DATABASE_PASSWORD@qwen-postgres:5432/qwen_api

OLLAMA_URL=http://host.docker.internal:11434

ADMIN_MASTER_KEY=YOUR_LONG_RANDOM_ADMIN_KEY
```

Generate a secure admin key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

---

# 🦙 Ollama Setup

Check Ollama:

```bash
systemctl status ollama
```

Check models:

```bash
curl http://127.0.0.1:11434/api/tags
```

Install the models if necessary:

```bash
ollama pull qwen3.5:0.8b
ollama pull qwen3-embedding:0.6b
```

Test Ollama:

```bash
curl http://127.0.0.1:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:0.8b",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": false
  }'
```

---

# 🐳 Docker

## 🔨 Build

```bash
docker build -t qwen-api .
```

Fresh build:

```bash
docker build --no-cache -t qwen-api .
```

## 🚀 Run

The API uses host port **8002** and container port **8000**.

```bash
docker run -d \
  --name qwen-api \
  --restart unless-stopped \
  --env-file .env \
  --network qwen-api_default \
  --add-host=host.docker.internal:host-gateway \
  -p 8002:8000 \
  qwen-api
```

Check:

```bash
docker ps --filter name=qwen-api
```

Expected:

```text
0.0.0.0:8002->8000/tcp
```

---

# 💚 Health Check

```bash
curl http://127.0.0.1:8002/health
```

Example:

```json
{
  "status": "ok",
  "model": "qwen3.5:0.8b",
  "embedding_model": "qwen3-embedding:0.6b",
  "embedding_dimensions": 1024
}
```

---

# 📚 API Documentation

Swagger UI:

```text
http://YOUR_SERVER_IP:8002/docs
```

OpenAPI:

```text
http://YOUR_SERVER_IP:8002/openapi.json
```

---

# 🔐 Authentication

There are two authentication levels.

### 👑 Admin Authentication

Used for API-key management.

```http
X-Admin-Key: YOUR_ADMIN_MASTER_KEY
```

### 🔑 Client Authentication

Used by normal API clients.

```http
Authorization: Bearer YOUR_API_KEY
```

---

# 🔑 API Key Management

## ➕ Create API Key

```bash
curl -X POST http://127.0.0.1:8002/v1/keys \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: YOUR_ADMIN_MASTER_KEY" \
  -d '{
    "name": "my-client"
  }'
```

Example:

```json
{
  "id": 1,
  "name": "my-client",
  "api_key": "qwen_...",
  "warning": "Save this API key now. It will not be shown again."
}
```

> ⚠️ **Important:** The complete API key is shown only once.

---

## 📋 List API Keys

```bash
curl http://127.0.0.1:8002/v1/keys \
  -H "X-Admin-Key: YOUR_ADMIN_MASTER_KEY"
```

Only metadata is returned.

The full secret key is never returned.

---

## 🚫 Revoke API Key

```bash
curl -X DELETE \
  http://127.0.0.1:8002/v1/keys/1 \
  -H "X-Admin-Key: YOUR_ADMIN_MASTER_KEY"
```

---

# 🤖 Chat Completions

Endpoint:

```text
POST /v1/chat/completions
```

Example:

```bash
curl -X POST http://127.0.0.1:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "qwen3.5:0.8b",
    "messages": [
      {
        "role": "user",
        "content": "Explain Docker in simple terms."
      }
    ],
    "stream": false,
    "temperature": 0.2,
    "max_tokens": 256
  }'
```

---

# 🌊 Streaming

Set:

```json
"stream": true
```

Example:

```bash
curl -N -X POST http://127.0.0.1:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "qwen3.5:0.8b",
    "messages": [
      {
        "role": "user",
        "content": "Tell me a very short story about a robot."
      }
    ],
    "stream": true,
    "temperature": 0.2,
    "max_tokens": 150
  }'
```

The API returns Server-Sent Events:

```text
data: {"object":"chat.completion.chunk", ...}

data: {"object":"chat.completion.chunk", ...}

data: {"object":"chat.completion.chunk", ...}

data: {"choices":[{"finish_reason":"stop"}]}

data: [DONE]
```

The client should concatenate:

```text
choices[0].delta.content
```

---

# 🔢 Embeddings

Endpoint:

```text
POST /v1/embeddings
```

Example:

```bash
curl -X POST http://127.0.0.1:8002/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "qwen3-embedding:0.6b",
    "input": "What is artificial intelligence?"
  }'
```

The response contains a **1024-dimensional vector**.

---

# 📦 Batch Embeddings

Up to **8 texts** can be submitted in one request.

```bash
curl -X POST http://127.0.0.1:8002/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "qwen3-embedding:0.6b",
    "input": [
      "What is AI?",
      "What is machine learning?",
      "What is deep learning?"
    ]
  }'
```

---

# 🚦 Rate Limiting

Current configuration:

```text
30 requests
per
60 seconds
per API key
```

When exceeded:

```text
HTTP 429 Too Many Requests
```

Response:

```json
{
  "detail": {
    "error": {
      "message": "Rate limit exceeded. Please try again later.",
      "type": "rate_limit_error",
      "code": "rate_limit_exceeded"
    }
  }
}
```

The response includes:

```http
Retry-After: 60
```

> 💡 The current limiter is intentionally in-memory and designed for a single FastAPI worker.

---

# 📏 Resource Limits

### 💬 Chat

| Limit | Value |
|---|---:|
| 📨 Maximum messages | `32` |
| 📝 Maximum chars/message | `16,000` |
| 📚 Maximum total input | `50,000` chars |
| 🎯 Maximum output tokens | `512` |

### 🔢 Embeddings

| Limit | Value |
|---|---:|
| 📦 Maximum batch | `8` |
| 📝 Maximum chars/text | `16,000` |
| 📚 Maximum total input | `50,000` chars |
| 📐 Dimensions | `1024` |

---

# 🧩 API Endpoints

| Method | Endpoint | 🔐 Auth | Description |
|:---:|---|:---:|---|
| `GET` | `/` | ❌ | API information |
| `GET` | `/health` | ❌ | Health check |
| `GET` | `/docs` | ❌ | Swagger UI |
| `GET` | `/openapi.json` | ❌ | OpenAPI schema |
| `POST` | `/v1/keys` | 👑 | Create API key |
| `GET` | `/v1/keys` | 👑 | List API keys |
| `DELETE` | `/v1/keys/{id}` | 👑 | Revoke API key |
| `GET` | `/v1/models` | 🔑 | List models |
| `POST` | `/v1/chat/completions` | 🔑 | Chat completion |
| `POST` | `/v1/embeddings` | 🔑 | Generate embeddings |

---

# 🐍 Python Example

```python
import requests

API_URL = "http://YOUR_SERVER_IP:8002"
API_KEY = "YOUR_API_KEY"

response = requests.post(
    f"{API_URL}/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "qwen3.5:0.8b",
        "messages": [
            {
                "role": "user",
                "content": "Hello! How are you?"
            }
        ],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 256,
    },
    timeout=300,
)

response.raise_for_status()

data = response.json()

print(data["choices"][0]["message"]["content"])
```

---

# 🟨 JavaScript Example

```javascript
const response = await fetch(
  "http://YOUR_SERVER_IP:8002/v1/chat/completions",
  {
    method: "POST",
    headers: {
      "Authorization": "Bearer YOUR_API_KEY",
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "qwen3.5:0.8b",
      messages: [
        {
          role: "user",
          content: "Hello!"
        }
      ],
      stream: false,
      temperature: 0.2,
      max_tokens: 256
    })
  }
);

const data = await response.json();

console.log(
  data.choices[0].message.content
);
```

---

# 🔄 OpenAI-Compatible Configuration

Applications that support a custom OpenAI-compatible endpoint can use:

```text
┌────────────────────────────────┐
│       OpenAI-Compatible        │
├────────────────────────────────┤
│ Base URL                       │
│ http://YOUR_SERVER_IP:8002/v1  │
│                                │
│ API Key                        │
│ YOUR_API_KEY                   │
│                                │
│ Chat Model                     │
│ qwen3.5:0.8b                   │
│                                │
│ Embedding Model                │
│ qwen3-embedding:0.6b           │
└────────────────────────────────┘
```

---

# 🔧 Configuration

Main configuration values:

```python
MODEL_NAME = "qwen3.5:0.8b"

EMBEDDING_MODEL_NAME = "qwen3-embedding:0.6b"

EMBEDDING_DIMENSIONS = 1024

MAX_MESSAGES = 32

MAX_MESSAGE_CHARS = 16000

MAX_TOTAL_INPUT_CHARS = 50000

MAX_OUTPUT_TOKENS = 512

MAX_EMBEDDING_BATCH = 8

MAX_EMBEDDING_CHARS = 16000

MAX_TOTAL_EMBEDDING_CHARS = 50000

RATE_LIMIT_REQUESTS = 30

RATE_LIMIT_WINDOW_SECONDS = 60

OLLAMA_TIMEOUT = 300

OLLAMA_KEEP_ALIVE = "30m"
```

---

# 🛠️ Development

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run locally:

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

# 🐳 Docker Management

### 📊 Status

```bash
docker ps --filter name=qwen-api
```

### 📜 Logs

```bash
docker logs qwen-api
```

### 👀 Follow logs

```bash
docker logs -f qwen-api
```

### 🔄 Restart

```bash
docker restart qwen-api
```

### 🛑 Stop

```bash
docker stop qwen-api
```

### ▶️ Start

```bash
docker start qwen-api
```

### 🔍 Inspect

```bash
docker inspect qwen-api
```

---

# 🔨 Deploy Code Changes

After modifying the application:

```bash
cd ~/qwen-api
```

Build:

```bash
docker build -t qwen-api .
```

Recreate **only the Qwen API** container:

```bash
docker rm -f qwen-api
```

```bash
docker run -d \
  --name qwen-api \
  --restart unless-stopped \
  --env-file .env \
  --network qwen-api_default \
  --add-host=host.docker.internal:host-gateway \
  -p 8002:8000 \
  qwen-api
```

Verify:

```bash
docker logs --tail 50 qwen-api
```

---

# 🩺 Troubleshooting

## ❌ API not starting

```bash
docker logs qwen-api
```

---

## ❌ Database connection

Check:

```bash
docker exec qwen-api env | grep DATABASE_URL
```

Check Qwen PostgreSQL:

```bash
docker ps --filter name=qwen-postgres
```

---

## ❌ Ollama connection

Host:

```bash
curl http://127.0.0.1:11434/api/tags
```

From Docker:

```bash
docker exec qwen-api python -c "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:11434/api/tags').read().decode())"
```

The container must have:

```text
--add-host=host.docker.internal:host-gateway
```

---

## ❌ Response is cut off

If:

```json
"finish_reason": "length"
```

increase:

```json
"max_tokens": 512
```

The current API maximum is:

```text
512
```

---

## ❌ HTTP 401

For normal API requests:

```http
Authorization: Bearer YOUR_API_KEY
```

For admin requests:

```http
X-Admin-Key: YOUR_ADMIN_MASTER_KEY
```

---

## ❌ HTTP 429

The API key exceeded:

```text
30 requests / 60 seconds
```

Wait and retry.

---

# 🔒 Security

### 🚨 Never commit secrets

Do not commit:

```text
.env
API keys
ADMIN_MASTER_KEY
database passwords
```

Recommended `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

### 🔐 Production

For Internet-facing deployments, use:

```text
🌍 Internet
    │
    ▼
🔒 HTTPS Reverse Proxy
    │
    ▼
🚀 Qwen API :8002
    │
    ├── 🦙 Ollama
    │
    └── 🐘 PostgreSQL
```

Avoid exposing PostgreSQL or Ollama directly to the public Internet.

---

# 📊 Service Ports

| Service | Port | Location |
|---|---:|---|
| 🚀 Qwen API | `8002` | Docker Host |
| ⚡ FastAPI | `8000` | Container |
| 🦙 Ollama | `11434` | Host |
| 🐘 PostgreSQL | `5432` | Docker |

---

# 🧪 Testing

### Health

```bash
curl http://127.0.0.1:8002/health
```

### Models

```bash
curl http://127.0.0.1:8002/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Chat

```bash
curl -X POST \
  http://127.0.0.1:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "qwen3.5:0.8b",
    "messages": [
      {
        "role": "user",
        "content": "What is 15 + 7?"
      }
    ],
    "stream": false,
    "temperature": 0.2,
    "max_tokens": 100
  }'
```

### Embeddings

```bash
curl -X POST \
  http://127.0.0.1:8002/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "qwen3-embedding:0.6b",
    "input": "Hello world"
  }'
```

---

# 🗂️ Project Structure

```text
qwen-api/
│
├── 📁 app/
│   ├── 🐍 main.py
│   ├── 🗄️ database.py
│   ├── 🔐 dependencies.py
│   ├── 👤 models.py
│   └── 🔑 security.py
│
├── 📁 logs/
│
├── 🐍 .venv/
│
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml
├── 📦 requirements.txt
├── 🔒 .env
├── 🚫 .gitignore
├── 📄 LICENSE
└── 📖 README.md
```

---

# 📌 API Summary

```text
                    🚀 QWEN API
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
       ▼                 ▼                 ▼
   💬 Chat           🌊 Stream         🔢 Embeddings
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                         ▼
                    🦙 Ollama
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        qwen3.5:0.8b       qwen3-embedding:0.6b
```

---

# ⭐ Why Qwen API?

- 🏠 **Self-hosted**
- 💰 **No external LLM API cost**
- 🔒 **Your prompts stay on your infrastructure**
- ⚡ **Fast local inference**
- 🧩 **OpenAI-compatible API**
- 🌊 **Streaming support**
- 🔢 **Embedding support**
- 🔑 **API-key authentication**
- 🚦 **Built-in rate limiting**
- 🐳 **Docker-ready**
- 📚 **Swagger documentation**

---

# 📜 License

This project is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

<p align="center">

### 🤖 Built with FastAPI + Ollama + Qwen

**🚀 Local AI • 🔐 Secure API • ⚡ Lightweight Infrastructure**

</p>