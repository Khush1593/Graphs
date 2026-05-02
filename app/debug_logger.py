from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class DebugRunLogger:
    base_dir: str = "debug"
    run_id: str = field(default_factory=lambda: uuid4().hex[:8])
    file_path: str = ""

    def __post_init__(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(self.base_dir, f"run_{timestamp}_{self.run_id}.md")
        self._write_line("# AI BI Agent Debug Log")
        self._write_line("")
        self._write_line(f"- run_id: {self.run_id}")
        self._write_line(f"- started_at: {datetime.now().isoformat(timespec='seconds')}")
        self._write_line("")

    def log_event(self, title: str, data) -> None:
        self._write_line(f"## {title}")
        self._write_line(f"- time: {datetime.now().isoformat(timespec='seconds')}")
        self._write_line("")
        self._write_line("```json")
        self._write_line(self._to_pretty_json(data))
        self._write_line("```")
        self._write_line("")

    def log_text(self, title: str, text: str) -> None:
        self._write_line(f"## {title}")
        self._write_line(f"- time: {datetime.now().isoformat(timespec='seconds')}")
        self._write_line("")
        self._write_line(text)
        self._write_line("")

    def _to_pretty_json(self, data) -> str:
        try:
            return json.dumps(data, indent=2, ensure_ascii=True, default=str)
        except TypeError:
            return json.dumps({"raw": str(data)}, indent=2, ensure_ascii=True)

    def _write_line(self, text: str) -> None:
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
