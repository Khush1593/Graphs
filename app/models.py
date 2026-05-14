"""SQLAlchemy ORM models and session factory for project state.

State that used to live in metadata.json and artifacts/*.json now lives in a
relational store. The connection string is pulled from DATABASE_URL; if it is
missing we fall back to a local SQLite file purely for developer convenience.
Models are written with PostgreSQL as the real target, so we use cross-dialect
types (String, Integer, JSON, DateTime) and rely on Alembic-style migrations
later rather than create_all in production.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


DEFAULT_SQLITE_URL = "sqlite:///local_dev.db"


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_SQLITE_URL)


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    project_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    last_modified: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    current_step: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    upload_filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    upload_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    debug_log_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    # Opaque key handed to StorageClient.get/resolve to fetch the raw CSV.
    raw_data_key: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    artifacts: Mapped[list["Artifact"]] = relationship(
        "Artifact",
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_artifact_project_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    # Artifacts hold whatever JSON-serializable payload the pipeline step
    # produced (dict, list, etc.); the ORM type hint is therefore Any.
    data: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True, default=None)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    project: Mapped[Project] = relationship("Project", back_populates="artifacts")


_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        # SQLite + Streamlit means several reruns may share threads.
        return {"connect_args": {"check_same_thread": False}, "future": True}
    return {"pool_pre_ping": True, "future": True}


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = get_database_url()
        _engine = create_engine(url, **_engine_kwargs(url))
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionFactory


def init_db() -> None:
    """Create tables if they don't exist. Safe to call at startup.

    For PostgreSQL we still call this for dev; production deployments should
    manage schema via Alembic migrations instead.
    """
    Base.metadata.create_all(bind=get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session. Commits on success, rolls back on exception."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
