from __future__ import annotations

from dataclasses import dataclass, field

from tablestores import TableMatch, TableStore


TABLE_CUE_KEYWORDS = {
    "比例",
    "比率",
    "負擔",
    "補助",
    "保費",
    "金額",
    "多少",
    "幾%",
    "%",
    "政府",
    "投保",
    "自付",
}

VECTOR_CUE_KEYWORDS = {
    "什麼是",
    "何謂",
    "定義",
    "流程",
    "如何",
    "怎麼",
    "原因",
    "為什麼",
    "規定",
    "資格",
    "條件",
}


@dataclass(slots=True)
class RouteDecision:
    mode: str
    reason: str
    confidence: float
    table_matches: list[TableMatch] = field(default_factory=list)


class QueryRouter:
    def __init__(self, *, table_store: TableStore) -> None:
        self._table_store = table_store

    def route(self, query: str) -> RouteDecision:
        normalized_query = self._normalize(query)
        table_matches = self._table_store.search(query, limit=3)
        top_score = table_matches[0].score if table_matches else 0.0

        has_table_cue = any(keyword in normalized_query for keyword in self._normalized(TABLE_CUE_KEYWORDS))
        has_vector_cue = any(keyword in normalized_query for keyword in self._normalized(VECTOR_CUE_KEYWORDS))

        if top_score >= 8.0:
            return RouteDecision(
                mode="table",
                reason="high table match score",
                confidence=min(0.98, 0.55 + (top_score / 20)),
                table_matches=table_matches,
            )

        if has_table_cue and top_score >= 3.0:
            return RouteDecision(
                mode="table",
                reason="table cue keyword with matched table rows",
                confidence=min(0.92, 0.5 + (top_score / 18)),
                table_matches=table_matches,
            )

        if has_table_cue and table_matches:
            return RouteDecision(
                mode="table",
                reason="table cue keyword with weak table match",
                confidence=0.6,
                table_matches=table_matches,
            )

        if has_vector_cue:
            return RouteDecision(
                mode="vector",
                reason="explanatory query pattern",
                confidence=0.75,
                table_matches=table_matches,
            )

        return RouteDecision(
            mode="vector",
            reason="default to vector retrieval",
            confidence=0.55,
            table_matches=table_matches,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return "".join(value.lower().split())

    @staticmethod
    def _normalized(values: set[str]) -> set[str]:
        return {"".join(value.lower().split()) for value in values}
