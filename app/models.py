from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

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
        String(128),
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

    memory_limit: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        index=True,
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

    model_access: Mapped[list["APIKeyModelAccess"]] = relationship(
        "APIKeyModelAccess",
        back_populates="api_key",
        cascade="all, delete-orphan",
    )

    usage_records: Mapped[list["APIUsage"]] = relationship(
        "APIUsage",
        back_populates="api_key",
        cascade="all, delete-orphan",
    )


class APIKeyModelAccess(Base):
    __tablename__ = "api_key_model_access"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    api_key_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "api_keys.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )

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
        server_default="0",
    )

    monthly_tokens_used: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    daily_reset_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    monthly_reset_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    api_key: Mapped["APIKey"] = relationship(
        "APIKey",
        back_populates="model_access",
    )


class APIUsage(Base):
    __tablename__ = "api_usage"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    api_key_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "api_keys.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    endpoint: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    completion_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    response_time_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )

    api_key: Mapped["APIKey"] = relationship(
        "APIKey",
        back_populates="usage_records",
    )
