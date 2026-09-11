from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    key_prefix: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )

    key_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
    )

    memory_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    model_access = relationship(
        "APIKeyModelAccess",
        back_populates="api_key",
        cascade="all, delete-orphan",
    )

    usage_records = relationship(
        "APIUsage",
        back_populates="api_key",
        cascade="all, delete-orphan",
    )


class APIKeyModelAccess(Base):
    __tablename__ = "api_key_model_access"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    api_key_id: Mapped[int] = mapped_column(
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    # NULL in database means unlimited.
    # API accepts the string "unlimited".
    requests_per_minute: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    daily_token_limit: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    monthly_token_limit: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    daily_tokens_used: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    monthly_tokens_used: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    daily_reset_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    monthly_reset_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    api_key = relationship(
        "APIKey",
        back_populates="model_access",
    )


class APIUsage(Base):
    __tablename__ = "api_usage"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    api_key_id: Mapped[int] = mapped_column(
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    endpoint: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
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

    response_time_ms: Mapped[float | None] = mapped_column(
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )

    api_key = relationship(
        "APIKey",
        back_populates="usage_records",
    )
