"""Project state persistence.

State (project metadata, step tracking, artifacts) lives in a relational
database accessed via SQLAlchemy. Raw uploaded files (CSVs) live in a
StorageClient — local volume today, S3 tomorrow. The ProjectStore here is the
single entry point that ties the two together; the rest of the app talks to
ProjectStore, never to os/pathlib or to a Session directly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data_engine import SchemaInfo
from app.models import Artifact, Project, init_db, session_scope
from app.storage import StorageClient, build_default_storage


ARTIFACT_NAMES = (
    "upload_signature",
    "schema",
    "semantic_hints",
    "understanding",
    "clarifications",
    "user_clarifications",
    "goals",
    "user_goals",
    "confirmation",
    "confirmation_accepted",
)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt else None


def schema_to_dict(schema: SchemaInfo) -> dict:
    return {
        "columns": list(schema.columns),
        "dtypes": dict(schema.dtypes),
        "date_column": schema.date_column,
        "numeric_column": schema.numeric_column,
        "category_column": schema.category_column,
    }


def schema_from_dict(data: dict) -> SchemaInfo:
    return SchemaInfo(
        columns=list(data.get("columns", [])),
        dtypes=dict(data.get("dtypes", {})),
        date_column=data.get("date_column"),
        numeric_column=data.get("numeric_column"),
        category_column=data.get("category_column"),
    )


@dataclass
class ProjectMeta:
    """Plain view object handed to the UI. Decoupled from the ORM row so
    Streamlit code never touches a detached Session."""

    project_id: str
    name: str
    created_at: str
    last_modified: str
    current_step: int = 1
    upload_filename: str | None = None
    upload_size: int | None = None
    debug_log_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_orm(cls, row: Project) -> "ProjectMeta":
        return cls(
            project_id=row.project_id,
            name=row.name,
            created_at=_iso(row.created_at) or "",
            last_modified=_iso(row.last_modified) or "",
            current_step=row.current_step,
            upload_filename=row.upload_filename,
            upload_size=row.upload_size,
            debug_log_path=row.debug_log_path,
        )


_META_FIELDS = {
    "name",
    "current_step",
    "upload_filename",
    "upload_size",
    "debug_log_path",
}


class ProjectStore:
    """Facade over the DB + blob storage. Public surface is unchanged from
    the legacy JSON-on-disk store so the UI keeps working."""

    def __init__(
        self,
        storage: StorageClient | None = None,
        *,
        auto_init_db: bool = True,
    ) -> None:
        self.storage: StorageClient = storage or build_default_storage()
        if auto_init_db:
            init_db()

    # ----- raw-data key helpers -------------------------------------------------
    @staticmethod
    def _raw_prefix(project_id: str) -> str:
        return f"{project_id}/raw_data"

    @staticmethod
    def _raw_key(project_id: str, filename: str) -> str:
        return f"{project_id}/raw_data/{filename}"

    # ----- projects -------------------------------------------------------------
    def create_project(self, name: str | None = None) -> ProjectMeta:
        project_id = uuid4().hex[:12]
        now = datetime.utcnow()
        with session_scope() as s:
            row = Project(
                project_id=project_id,
                name=(name or "").strip() or f"project_{project_id}",
                created_at=now,
                last_modified=now,
                current_step=1,
            )
            s.add(row)
            s.flush()
            return ProjectMeta.from_orm(row)

    def list_projects(self) -> list[ProjectMeta]:
        with session_scope() as s:
            rows = s.execute(
                select(Project).order_by(Project.last_modified.desc())
            ).scalars().all()
            return [ProjectMeta.from_orm(r) for r in rows]

    def get_meta(self, project_id: str) -> ProjectMeta | None:
        with session_scope() as s:
            row = s.get(Project, project_id)
            return ProjectMeta.from_orm(row) if row else None

    def update_meta(
        self,
        project_id: str,
        bump_modified: bool = True,
        **fields: Any,
    ) -> ProjectMeta:
        with session_scope() as s:
            row = s.get(Project, project_id)
            if row is None:
                raise ValueError(f"Project {project_id} not found")
            for key, value in fields.items():
                if key in _META_FIELDS:
                    setattr(row, key, value)
            if bump_modified:
                row.last_modified = datetime.utcnow()
            s.flush()
            return ProjectMeta.from_orm(row)

    def rename_project(self, project_id: str, new_name: str) -> ProjectMeta:
        clean = (new_name or "").strip()
        if not clean:
            raise ValueError("New project name must not be empty.")
        return self.update_meta(project_id, name=clean)

    def delete_project(self, project_id: str) -> None:
        with session_scope() as s:
            row = s.get(Project, project_id)
            if row is not None:
                s.delete(row)
        # Blob deletion lives outside the DB transaction; if it fails the row
        # is already gone, which matches the prior on-disk behaviour.
        self.storage.delete_prefix(project_id)

    # ----- artifacts ------------------------------------------------------------
    def save_artifact(self, project_id: str, name: str, data: Any) -> None:
        now = datetime.utcnow()
        with session_scope() as s:
            self._upsert_artifact(s, project_id, name, data, now)
            row = s.get(Project, project_id)
            if row is not None:
                row.last_modified = now

    @staticmethod
    def _upsert_artifact(
        s: Session,
        project_id: str,
        name: str,
        data: Any,
        now: datetime,
    ) -> None:
        existing = s.execute(
            select(Artifact).where(
                Artifact.project_id == project_id, Artifact.name == name
            )
        ).scalar_one_or_none()
        if existing is None:
            s.add(Artifact(project_id=project_id, name=name, data=data, updated_at=now))
        else:
            existing.data = data
            existing.updated_at = now

    def load_artifact(self, project_id: str, name: str) -> Any | None:
        with session_scope() as s:
            row = s.execute(
                select(Artifact).where(
                    Artifact.project_id == project_id, Artifact.name == name
                )
            ).scalar_one_or_none()
            return row.data if row else None

    def has_artifact(self, project_id: str, name: str) -> bool:
        with session_scope() as s:
            row = s.execute(
                select(Artifact.id).where(
                    Artifact.project_id == project_id, Artifact.name == name
                )
            ).first()
            return row is not None

    def delete_artifact(self, project_id: str, name: str) -> None:
        with session_scope() as s:
            row = s.execute(
                select(Artifact).where(
                    Artifact.project_id == project_id, Artifact.name == name
                )
            ).scalar_one_or_none()
            if row is not None:
                s.delete(row)

    def clear_artifacts(self, project_id: str) -> None:
        with session_scope() as s:
            rows = s.execute(
                select(Artifact).where(Artifact.project_id == project_id)
            ).scalars().all()
            for row in rows:
                s.delete(row)

    # ----- raw uploaded data ----------------------------------------------------
    def save_raw_data(
        self,
        project_id: str,
        file_bytes: bytes,
        filename: str,
    ) -> str:
        # A project always has exactly one source CSV: wipe prior files first.
        self.storage.delete_prefix(self._raw_prefix(project_id))
        key = self._raw_key(project_id, filename)
        self.storage.put(key, file_bytes)
        with session_scope() as s:
            row = s.get(Project, project_id)
            if row is None:
                raise ValueError(f"Project {project_id} not found")
            row.raw_data_key = key
            row.upload_filename = filename
            row.upload_size = len(file_bytes)
            row.last_modified = datetime.utcnow()
        return self.storage.resolve(key)

    def load_raw_data(self, project_id: str) -> tuple[bytes, str] | None:
        with session_scope() as s:
            row = s.get(Project, project_id)
            key = row.raw_data_key if row else None
            filename = row.upload_filename if row else None
        if not key or not self.storage.exists(key):
            return None
        data = self.storage.get(key)
        return data, (filename or key.rsplit("/", 1)[-1])

    def has_raw_data(self, project_id: str) -> bool:
        with session_scope() as s:
            row = s.get(Project, project_id)
            key = row.raw_data_key if row else None
        return bool(key and self.storage.exists(key))

    def raw_data_path(self, project_id: str) -> str | None:
        """Return a backend-resolved identifier (local path / S3 URI) for the
        raw CSV. data_engine and downstream tooling should call this rather
        than constructing paths themselves."""
        with session_scope() as s:
            row = s.get(Project, project_id)
            key = row.raw_data_key if row else None
        return self.storage.resolve(key) if key else None
