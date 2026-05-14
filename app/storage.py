"""Storage abstraction for raw binary blobs (CSV uploads, etc.).

Business logic should depend on the StorageClient interface, not on os/pathlib
directly. Today the only implementation is LocalVolumeStorage; tomorrow we can
add S3Storage without touching persistence.py or the data ingestion path.
"""
from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from typing import Iterable


class StorageClient(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes) -> str: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None: ...

    @abstractmethod
    def list_prefix(self, prefix: str) -> list[str]: ...

    @abstractmethod
    def resolve(self, key: str) -> str:
        """Return an identifier callers can hand to downstream tools.

        For local storage this is the absolute filesystem path; for S3 it
        would be an s3:// URI or a presigned URL.
        """


class LocalVolumeStorage(StorageClient):
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _abs(self, key: str) -> str:
        # Reject traversal attempts; keys are treated as relative POSIX-ish paths.
        norm = os.path.normpath(key).lstrip(os.sep)
        if norm.startswith(".."):
            raise ValueError(f"Invalid storage key: {key!r}")
        return os.path.join(self.root, norm)

    def put(self, key: str, data: bytes) -> str:
        path = self._abs(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return key

    def get(self, key: str) -> bytes:
        with open(self._abs(key), "rb") as f:
            return f.read()

    def exists(self, key: str) -> bool:
        return os.path.isfile(self._abs(key))

    def delete(self, key: str) -> None:
        path = self._abs(key)
        if os.path.isfile(path):
            os.remove(path)

    def delete_prefix(self, prefix: str) -> None:
        path = self._abs(prefix)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    def list_prefix(self, prefix: str) -> list[str]:
        path = self._abs(prefix)
        if not os.path.isdir(path):
            return []
        out: list[str] = []
        for root, _dirs, files in os.walk(path):
            for name in files:
                rel = os.path.relpath(os.path.join(root, name), self.root)
                out.append(rel.replace(os.sep, "/"))
        return sorted(out)

    def resolve(self, key: str) -> str:
        return self._abs(key)


def build_default_storage() -> StorageClient:
    """Factory used by persistence.py when no client is injected."""
    root = os.environ.get("LOCAL_STORAGE_ROOT", "projects")
    return LocalVolumeStorage(root)
