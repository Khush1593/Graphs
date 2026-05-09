from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from app.data_engine import SchemaInfo


PROJECTS_ROOT = "projects"

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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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
    def from_dict(cls, data: dict) -> "ProjectMeta":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


class ProjectStore:
    def __init__(self, root: str = PROJECTS_ROOT) -> None:
        self.root = root
        _ensure_dir(self.root)

    def _project_dir(self, project_id: str) -> str:
        return os.path.join(self.root, project_id)

    def _meta_path(self, project_id: str) -> str:
        return os.path.join(self._project_dir(project_id), "metadata.json")

    def _artifacts_dir(self, project_id: str) -> str:
        return os.path.join(self._project_dir(project_id), "artifacts")

    def _raw_dir(self, project_id: str) -> str:
        return os.path.join(self._project_dir(project_id), "raw_data")

    def _artifact_path(self, project_id: str, name: str) -> str:
        return os.path.join(self._artifacts_dir(project_id), f"{name}.json")

    def create_project(self, name: str | None = None) -> ProjectMeta:
        project_id = uuid4().hex[:12]
        now = _now_iso()
        meta = ProjectMeta(
            project_id=project_id,
            name=(name or "").strip() or f"project_{project_id}",
            created_at=now,
            last_modified=now,
            current_step=1,
        )
        _ensure_dir(self._project_dir(project_id))
        _ensure_dir(self._artifacts_dir(project_id))
        _ensure_dir(self._raw_dir(project_id))
        self._save_meta(meta)
        return meta

    def list_projects(self) -> list[ProjectMeta]:
        if not os.path.isdir(self.root):
            return []
        projects: list[ProjectMeta] = []
        for entry in os.listdir(self.root):
            meta_path = os.path.join(self.root, entry, "metadata.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path) as f:
                        projects.append(ProjectMeta.from_dict(json.load(f)))
                except Exception:
                    continue
        projects.sort(key=lambda p: p.last_modified, reverse=True)
        return projects

    def get_meta(self, project_id: str) -> ProjectMeta | None:
        path = self._meta_path(project_id)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return ProjectMeta.from_dict(json.load(f))

    def _save_meta(self, meta: ProjectMeta) -> None:
        _ensure_dir(self._project_dir(meta.project_id))
        with open(self._meta_path(meta.project_id), "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

    def update_meta(
        self,
        project_id: str,
        bump_modified: bool = True,
        **fields,
    ) -> ProjectMeta:
        meta = self.get_meta(project_id)
        if meta is None:
            raise ValueError(f"Project {project_id} not found")
        for key, value in fields.items():
            if hasattr(meta, key):
                setattr(meta, key, value)
        if bump_modified:
            meta.last_modified = _now_iso()
        self._save_meta(meta)
        return meta

    def rename_project(self, project_id: str, new_name: str) -> ProjectMeta:
        clean = (new_name or "").strip()
        if not clean:
            raise ValueError("New project name must not be empty.")
        return self.update_meta(project_id, name=clean)

    def delete_project(self, project_id: str) -> None:
        path = self._project_dir(project_id)
        if os.path.isdir(path):
            shutil.rmtree(path)

    def save_artifact(self, project_id: str, name: str, data: Any) -> None:
        _ensure_dir(self._artifacts_dir(project_id))
        with open(self._artifact_path(project_id, name), "w") as f:
            json.dump(data, f, indent=2, default=str)
        self.update_meta(project_id)

    def load_artifact(self, project_id: str, name: str) -> Any | None:
        path = self._artifact_path(project_id, name)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return json.load(f)

    def has_artifact(self, project_id: str, name: str) -> bool:
        return os.path.isfile(self._artifact_path(project_id, name))

    def delete_artifact(self, project_id: str, name: str) -> None:
        path = self._artifact_path(project_id, name)
        if os.path.isfile(path):
            os.remove(path)

    def clear_artifacts(self, project_id: str) -> None:
        artifacts_dir = self._artifacts_dir(project_id)
        if os.path.isdir(artifacts_dir):
            shutil.rmtree(artifacts_dir)
        _ensure_dir(artifacts_dir)

    def save_raw_data(
        self,
        project_id: str,
        file_bytes: bytes,
        filename: str,
    ) -> str:
        raw_dir = self._raw_dir(project_id)
        # Replace any prior raw data so projects always have a single source CSV.
        if os.path.isdir(raw_dir):
            shutil.rmtree(raw_dir)
        _ensure_dir(raw_dir)
        path = os.path.join(raw_dir, filename)
        with open(path, "wb") as f:
            f.write(file_bytes)
        return path

    def load_raw_data(self, project_id: str) -> tuple[bytes, str] | None:
        raw_dir = self._raw_dir(project_id)
        if not os.path.isdir(raw_dir):
            return None
        files = sorted(os.listdir(raw_dir))
        if not files:
            return None
        filename = files[0]
        with open(os.path.join(raw_dir, filename), "rb") as f:
            return f.read(), filename

    def has_raw_data(self, project_id: str) -> bool:
        return self.load_raw_data(project_id) is not None
