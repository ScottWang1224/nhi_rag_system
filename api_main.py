from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
TEMPLATES_PATH = PROJECT_ROOT / "templates"
STATIC_PATH = PROJECT_ROOT / "static"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from api import ChatRequest, ChatResponse, ReferenceLink
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    config, service = build_service(PROJECT_ROOT)
    app.state.config = config
    app.state.service = service
    yield


app = FastAPI(title="NHI RAG API", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(TEMPLATES_PATH / "index.html")


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    config: AppConfig = request.app.state.config
    return {
        "status": "ok",
        "collection_name": config.collection_name,
        "vectordb_path": str(config.vectordb_path),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    service: RAGService = request.app.state.service
    config: AppConfig = request.app.state.config

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        result = service.answer_question(query, top_k=payload.top_k or config.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG request failed: {exc}") from exc

    return ChatResponse(
        query=result.query,
        answer=result.answer,
        references=_serialize_references(result.retrieved_chunks),
    )
