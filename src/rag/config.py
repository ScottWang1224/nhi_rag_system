from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    vectordb_path: Path
    chunks_path: Path
    collection_name: str
    embedding_model: str
    answer_model: str
    top_k: int
    openai_api_key: str

    @classmethod
    def from_env(cls, project_root: Path) -> "AppConfig":
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("找不到 OPENAI_API_KEY，請確認環境變數或 .env 已設定。")

        return cls(
            project_root=project_root,
            vectordb_path=project_root / "data" / "vectordb" / "chroma",
            chunks_path=project_root / "data" / "processed" / "chunks.json",
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "nhi_rag_chunks"),
            embedding_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            answer_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            top_k=int(os.getenv("RAG_TOP_K", "3")),
            openai_api_key=api_key,
        )
