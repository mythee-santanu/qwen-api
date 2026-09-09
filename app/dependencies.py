from datetime import datetime

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .database import get_db
from .models import APIKey
from .security import hash_api_key


bearer_scheme = HTTPBearer(
    auto_error=False,
)


def get_current_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
    db: Session = Depends(get_db),
) -> APIKey:

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme",
        )

    api_key = credentials.credentials

    key_hash = hash_api_key(api_key)

    db_key = (
        db.query(APIKey)
        .filter(APIKey.key_hash == key_hash)
        .first()
    )

    if db_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    if not db_key.active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has been revoked",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    db_key.last_used_at = datetime.utcnow()
    db.commit()

    return db_key