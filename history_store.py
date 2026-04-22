"""历史记录存储模块。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List

HISTORY_SCHEMA_VERSION = 1


class HistoryStore:
    def __init__(self, history_file: Path, output_dir: Path):
        self.history_file = history_file
        self.output_dir = output_dir

    def load(self) -> List[Dict[str, Any]]:
        if not self.history_file.exists():
            return []

        try:
            with self.history_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, JSONDecodeError):
            return []

        if not isinstance(data, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            normalized.append(self._normalize_item(item))
        return normalized

    def save(self, history: List[Dict[str, Any]]) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.history_file.with_suffix(self.history_file.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.history_file)

    def append(self, history: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        new_meta = self._normalize_item(meta)
        history.append(new_meta)
        self.save(history)
        return history

    def remove(self, history: List[Dict[str, Any]], index: int) -> List[Dict[str, Any]]:
        if 0 <= index < len(history):
            del history[index]
            self.save(history)
        return history

    def clear(self) -> List[Dict[str, Any]]:
        self.save([])
        return []

    def resolve_audio_path(self, audio_path: str) -> Path:
        p = Path(audio_path)
        if p.is_absolute():
            return p
        return (self.output_dir.parent / p).resolve()

    def _normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(item)
        if "schema_version" not in normalized:
            normalized["schema_version"] = HISTORY_SCHEMA_VERSION
        if "created_at" not in normalized:
            normalized["created_at"] = datetime.now().isoformat(timespec="seconds")
        if "audio_path" in normalized:
            normalized["audio_path"] = self._to_relative_output_path(str(normalized["audio_path"]))
        return normalized

    def _to_relative_output_path(self, path_str: str) -> str:
        p = Path(path_str)
        try:
            if p.is_absolute():
                return str(p.relative_to(self.output_dir.parent.resolve()))
            return str(p)
        except ValueError:
            return str(p)
