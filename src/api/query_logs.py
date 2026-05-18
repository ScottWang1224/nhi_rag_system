from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class QueryLogStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")

    def list_records(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []

        records: list[dict[str, Any]] = []
        with self._path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    records.append(
                        {
                            "request_id": f"invalid_line_{line_number}",
                            "status": "invalid_log_line",
                            "error": str(exc),
                            "raw": line,
                        }
                    )
                    continue
                records.append(record)

        return records

    def update_review(
        self,
        request_id: str,
        *,
        human_rating: str | None,
        human_note: str | None,
    ) -> dict[str, Any] | None:
        records = self.list_records()
        updated_record: dict[str, Any] | None = None

        for record in records:
            if record.get("request_id") != request_id:
                continue
            record["human_rating"] = human_rating
            record["human_note"] = human_note
            updated_record = record
            break

        if updated_record is None:
            return None

        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            for record in records:
                json.dump(record, file, ensure_ascii=False)
                file.write("\n")
        temp_path.replace(self._path)

        return updated_record

    def update_feedback(
        self,
        request_id: str,
        *,
        user_feedback: str,
        user_feedback_at: str,
    ) -> dict[str, Any] | None:
        records = self.list_records()
        updated_record: dict[str, Any] | None = None

        for record in records:
            if record.get("request_id") != request_id:
                continue
            record["user_feedback"] = user_feedback
            record["user_feedback_at"] = user_feedback_at
            updated_record = record
            break

        if updated_record is None:
            return None

        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            for record in records:
                json.dump(record, file, ensure_ascii=False)
                file.write("\n")
        temp_path.replace(self._path)

        return updated_record
