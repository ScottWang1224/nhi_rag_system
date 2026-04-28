from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TABLE_QUERY_KEYWORDS = {
    "比例",
    "比率",
    "負擔",
    "補助",
    "保費",
    "金額",
    "多少",
    "幾%",
    "%",
    "誰付",
    "政府",
    "投保",
}


@dataclass(slots=True)
class TableDefinition:
    table_id: str
    table_type: str
    title: str
    source: str
    url: str
    rows: list[dict[str, Any]]
    field_description: dict[str, str]
    general_rules: dict[str, Any]
    search_blob: str


@dataclass(slots=True)
class TableMatch:
    table_id: str
    title: str
    url: str
    source: str
    score: float
    matched_rows: list[dict[str, Any]]
    matched_fields: list[str]
    general_rules: dict[str, Any]


class TableStore:
    def __init__(self, table_json_path: Path) -> None:
        self._table_json_path = table_json_path
        self._tables = self._load_tables(table_json_path)

    @property
    def tables(self) -> list[TableDefinition]:
        return self._tables

    def search(self, query: str, *, limit: int = 3, row_limit: int = 3) -> list[TableMatch]:
        normalized_query = self._normalize(query)
        terms = self._extract_terms(query)
        has_table_keyword = any(keyword in normalized_query for keyword in self._normalized_keywords())

        matches: list[TableMatch] = []
        for table in self._tables:
            table_score = self._score_text(normalized_query, table.search_blob, terms)
            title_score = self._score_text(normalized_query, self._normalize(table.title), terms) * 1.5
            field_hits = self._matched_fields(normalized_query, table.field_description)
            row_matches = self._match_rows(normalized_query, terms, table.rows, row_limit=row_limit)

            total_score = table_score + title_score + (len(field_hits) * 1.5)
            if row_matches:
                total_score += row_matches[0][0] * 1.2
            if has_table_keyword:
                total_score += 1.0

            if total_score <= 0:
                continue

            matches.append(
                TableMatch(
                    table_id=table.table_id,
                    title=table.title,
                    url=table.url,
                    source=table.source,
                    score=round(total_score, 2),
                    matched_rows=[row for _, row in row_matches],
                    matched_fields=field_hits,
                    general_rules=table.general_rules,
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]

    def format_context(self, match: TableMatch) -> str:
        lines = [
            f"table_id: {match.table_id}",
            f"title: {match.title}",
            f"url: {match.url}",
        ]

        if match.matched_fields:
            lines.append(f"matched_fields: {', '.join(match.matched_fields)}")

        if match.matched_rows:
            lines.append("matched_rows:")
            for index, row in enumerate(match.matched_rows, start=1):
                lines.append(f"- row_{index}: {json.dumps(row, ensure_ascii=False)}")

        if match.general_rules:
            lines.append(f"general_rules: {json.dumps(match.general_rules, ensure_ascii=False)}")

        return "\n".join(lines)

    def _load_tables(self, table_json_path: Path) -> list[TableDefinition]:
        if not table_json_path.exists():
            raise FileNotFoundError(f"Table data not found: {table_json_path}")

        with table_json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        tables: list[TableDefinition] = []
        for raw_table in payload.get("tables", []):
            title = str(raw_table.get("title", ""))
            table_type = str(raw_table.get("type", ""))
            field_description = raw_table.get("field_description", {}) or {}
            general_rules = raw_table.get("general_rules", {}) or {}
            rows = raw_table.get("rows", []) or []
            search_blob = self._build_search_blob(title, table_type, field_description, general_rules, rows)

            tables.append(
                TableDefinition(
                    table_id=str(raw_table.get("table_id", "")),
                    table_type=table_type,
                    title=title,
                    source=str(raw_table.get("source", "")),
                    url=str(raw_table.get("url", "")),
                    rows=rows,
                    field_description={str(key): str(value) for key, value in field_description.items()},
                    general_rules=general_rules,
                    search_blob=search_blob,
                )
            )

        return tables

    @staticmethod
    def _build_search_blob(
        title: str,
        table_type: str,
        field_description: dict[str, Any],
        general_rules: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> str:
        parts = [title, table_type]
        parts.extend(str(value) for value in field_description.values())
        parts.extend(str(value) for value in general_rules.values())
        for row in rows:
            parts.extend(str(value) for value in row.values())
        return TableStore._normalize(" ".join(parts))

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", "", value).lower()

    @staticmethod
    def _extract_terms(query: str) -> list[str]:
        raw_terms = re.findall(r"[A-Za-z0-9%]+|[\u4e00-\u9fff]{1,}", query)
        terms: list[str] = []
        for term in raw_terms:
            normalized = TableStore._normalize(term)
            if not normalized:
                continue
            terms.append(normalized)

            if re.fullmatch(r"[\u4e00-\u9fff]{4,}", term):
                terms.extend(TableStore._normalize(term[index:index + 2]) for index in range(len(term) - 1))

        unique_terms = list(dict.fromkeys(terms))
        return [term for term in unique_terms if len(term) >= 2 or term.isdigit() or "%" in term]

    @staticmethod
    def _score_text(normalized_query: str, haystack: str, terms: list[str]) -> float:
        score = 0.0
        if normalized_query and normalized_query in haystack:
            score += 6.0

        for term in terms:
            if term in haystack:
                score += min(3.0, max(1.0, len(term) / 2))
        return score

    @staticmethod
    def _matched_fields(normalized_query: str, field_description: dict[str, str]) -> list[str]:
        hits: list[str] = []
        for field_name, description in field_description.items():
            normalized_description = TableStore._normalize(description)
            if normalized_description and normalized_description in normalized_query:
                hits.append(field_name)
                continue

            if TableStore._normalize(field_name) in normalized_query:
                hits.append(field_name)
        return hits

    @staticmethod
    def _match_rows(
        normalized_query: str,
        terms: list[str],
        rows: list[dict[str, Any]],
        *,
        row_limit: int,
    ) -> list[tuple[float, dict[str, Any]]]:
        scored_rows: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            row_text = TableStore._normalize(" ".join(str(value) for value in row.values()))
            row_score = TableStore._score_text(normalized_query, row_text, terms)
            if row_score > 0:
                scored_rows.append((row_score, row))

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return scored_rows[:row_limit]

    @staticmethod
    def _normalized_keywords() -> set[str]:
        return {TableStore._normalize(keyword) for keyword in TABLE_QUERY_KEYWORDS}
