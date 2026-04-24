from __future__ import annotations

from pathlib import Path

from rag.config import AppConfig
from rag.service import RAGService
from vectorstores import ChromaRetriever


def build_service(project_root: Path) -> tuple[AppConfig, RAGService]:
    config = AppConfig.from_env(project_root)

    if not config.vectordb_path.exists():
        raise FileNotFoundError(
            f"Chroma database not found at {config.vectordb_path}. "
            "Run scripts/index_chroma.py first."
        )

    retriever = ChromaRetriever(
        vectordb_path=config.vectordb_path,
        collection_name=config.collection_name,
        api_key=config.openai_api_key,
        embedding_model=config.embedding_model,
    )
    service = RAGService(
        retriever=retriever,
        api_key=config.openai_api_key,
        answer_model=config.answer_model,
        top_k=config.top_k,
    )
    return config, service
