from __future__ import annotations

import sys
import time
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
TEMPLATES_PATH = PROJECT_ROOT / "templates"
STATIC_PATH = PROJECT_ROOT / "static"
QUERY_LOGS_VIEWER_PATH = PROJECT_ROOT / "eval" / "query_logs_viewer.html"
QUERY_LOGS_PATH = PROJECT_ROOT / "data" / "logs" / "query_logs.jsonl"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from api import ChatRequest, ChatResponse, QueryLogReviewUpdate, ReferenceLink
from api.query_logs import QueryLogStore
from rag import AppConfig, RAGService, build_service


def _serialize_references(result_sources) -> list[ReferenceLink]:
    seen: set[tuple[str, str]] = set()
    references: list[ReferenceLink] = []

    for chunk in result_sources:
        metadata = chunk.metadata
        url = str(metadata.get("url") or "").strip()
        if not url:
            continue

        title = str(metadata.get("question") or metadata.get("source") or url).strip()
        source = metadata.get("source")
        dedupe_key = (title, url)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        references.append(
            ReferenceLink(
                rank=len(references) + 1,
                title=title,
                url=url,
                source=source,
            )
        )

    return references


def _serialize_answer_references(result_references) -> list[ReferenceLink]:
    return [
        ReferenceLink(
            rank=index,
            title=reference.title,
            url=reference.url,
            source=reference.source_type,
        )
        for index, reference in enumerate(result_references, start=1)
    ]


def _now_taipei_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")


def _serialize_chunks(chunks) -> list[dict]:
    serialized = []
    for chunk in chunks:
        metadata = chunk.metadata
        serialized.append(
            {
                "rank": chunk.rank,
                "doc_id": str(metadata.get("doc_id", "")).strip(),
                "source": str(metadata.get("source", "")).strip(),
                "url": str(metadata.get("url", "")).strip(),
                "question": str(metadata.get("question", "")).strip(),
                "chunk_type": metadata.get("chunk_type"),
                "length_type": metadata.get("length_type"),
                "chunk_index": metadata.get("chunk_index"),
                "chunk_total": metadata.get("chunk_total"),
                "distance": chunk.distance,
            }
        )
    return serialized


def _serialize_table_matches(table_matches) -> list[dict]:
    return [
        {
            "rank": index,
            "table_id": match.table_id,
            "title": match.title,
            "url": match.url,
            "source": match.source,
            "score": match.score,
            "matched_rows": match.matched_rows,
            "matched_fields": match.matched_fields,
            "general_rules": match.general_rules,
        }
        for index, match in enumerate(table_matches, start=1)
    ]


def _reference_links_to_dicts(references: list[ReferenceLink]) -> list[dict]:
    return [
        {
            "rank": reference.rank,
            "title": reference.title,
            "url": reference.url,
            "source": reference.source,
        }
        for reference in references
    ]


def _build_success_log_record(
    *,
    request_id: str,
    query: str,
    top_k: int,
    latency_ms: int,
    result,
    references: list[ReferenceLink],
) -> dict:
    return {
        "request_id": request_id,
        "created_at": _now_taipei_iso(),
        "query": query,
        "top_k": top_k,
        "answer": result.answer,
        "route_mode": result.route_mode,
        "route_reason": result.route_reason,
        "route_confidence": result.route_confidence,
        "retrieved_chunks": _serialize_chunks(result.retrieved_chunks),
        "table_matches": _serialize_table_matches(result.table_matches),
        "references": _reference_links_to_dicts(references),
        "latency_ms": latency_ms,
        "status": "success",
        "error": None,
        "human_rating": None,
        "human_note": None,
    }


def _build_error_log_record(
    *,
    request_id: str,
    query: str,
    top_k: int | None,
    latency_ms: int,
    error: str,
) -> dict:
    return {
        "request_id": request_id,
        "created_at": _now_taipei_iso(),
        "query": query,
        "top_k": top_k,
        "answer": "",
        "route_mode": "error",
        "route_reason": "",
        "route_confidence": None,
        "retrieved_chunks": [],
        "table_matches": [],
        "references": [],
        "latency_ms": latency_ms,
        "status": "error",
        "error": error,
        "human_rating": None,
        "human_note": None,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    config, service = build_service(PROJECT_ROOT)
    app.state.config = config
    app.state.service = service
    app.state.query_log_store = QueryLogStore(QUERY_LOGS_PATH)
    yield


app = FastAPI(title="NHI RAG API", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(TEMPLATES_PATH / "index.html")


@app.get("/query-logs")
async def query_logs_viewer() -> FileResponse:
    return FileResponse(QUERY_LOGS_VIEWER_PATH)


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    config: AppConfig = request.app.state.config
    return {
        "status": "ok",
        "collection_name": config.collection_name,
        "vectordb_path": str(config.vectordb_path),
    }


@app.get("/api/query-logs")
async def query_logs(request: Request) -> dict:
    log_store: QueryLogStore = request.app.state.query_log_store
    records = log_store.list_records()
    return {
        "log_path": str(log_store.path),
        "total": len(records),
        "records": list(reversed(records)),
    }


@app.patch("/api/query-logs/{request_id}/review")
async def update_query_log_review(
    request_id: str,
    payload: QueryLogReviewUpdate,
    request: Request,
) -> dict:
    log_store: QueryLogStore = request.app.state.query_log_store
    updated = log_store.update_review(
        request_id,
        human_rating=payload.human_rating,
        human_note=payload.human_note,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Query log record not found.")
    return {"record": updated}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    service: RAGService = request.app.state.service
    config: AppConfig = request.app.state.config
    log_store: QueryLogStore = request.app.state.query_log_store

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    request_id = str(uuid4())
    started_at = time.perf_counter()
    top_k = payload.top_k or config.top_k

    try:
        result = service.answer_question(query, top_k=top_k)
        references = (
            _serialize_answer_references(result.references)
            if result.references
            else _serialize_references(result.retrieved_chunks)
        )
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        log_store.append(
            _build_success_log_record(
                request_id=request_id,
                query=query,
                top_k=top_k,
                latency_ms=latency_ms,
                result=result,
                references=references,
            )
        )
    except FileNotFoundError as exc:
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        log_store.append(
            _build_error_log_record(
                request_id=request_id,
                query=query,
                top_k=top_k,
                latency_ms=latency_ms,
                error=str(exc),
            )
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        log_store.append(
            _build_error_log_record(
                request_id=request_id,
                query=query,
                top_k=top_k,
                latency_ms=latency_ms,
                error=f"RAG request failed: {exc}",
            )
        )
        raise HTTPException(status_code=500, detail=f"RAG request failed: {exc}") from exc

    return ChatResponse(
        request_id=request_id,
        query=result.query,
        answer=result.answer,
        references=references,
    )
