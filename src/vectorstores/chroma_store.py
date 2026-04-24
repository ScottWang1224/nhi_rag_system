from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


@dataclass(slots=True)
class RetrievedChunk:
    rank: int
    document: str
    metadata: dict[str, Any]
    distance: float | None


class ChromaRetriever:
    def __init__(
        self,
        *,
        vectordb_path: Path,
        collection_name: str,
        api_key: str,
        embedding_model: str,
    ) -> None:
        self._client = chromadb.PersistentClient(path=str(vectordb_path))
        self._embedding_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embedding_model,
        )
        self._collection = self._client.get_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )

    def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for index, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances),
            start=1,
        ):
            retrieved.append(
                RetrievedChunk(
                    rank=index,
                    document=document,
                    metadata=metadata or {},
                    distance=distance,
                )
            )
        return retrieved
