import json
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
VECTORDB_PATH = PROJECT_ROOT / "data" / "vectordb" / "chroma"
COLLECTION_NAME = "nhi_rag_chunks"

BATCH_SIZE = 32
EMBED_MODEL = "text-embedding-3-small"


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("找不到 OPENAI_API_KEY，請確認 .env 設定")

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"找不到 chunks.json: {CHUNKS_PATH}")

    VECTORDB_PATH.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client = chromadb.PersistentClient(path=str(VECTORDB_PATH))

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL,
    )

    # 若已存在同名 collection，直接取用；不存在就建立
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )

    ids = []
    documents = []
    metadatas = []

    for c in chunks:
        ids.append(c["chunk_id"])
        documents.append(c["content"])
        metadatas.append({
            "doc_id": c["doc_id"],
            "source": c["source"],
            "url": c["url"],
            "question": c["question"],
            "chunk_type": c["chunk_type"],
            "length_type": c["length_type"],
            "chunk_index": c["chunk_index"],
            "chunk_total": c["chunk_total"],
            "char_len": c["char_len"],
        })

    total = len(ids)

    # 先刪除同 id，避免重跑時重複
    # Chroma collection 支援 delete by ids
    for batch_ids in chunked(ids, BATCH_SIZE):
        try:
            collection.delete(ids=batch_ids)
        except Exception:
            pass

    # 批次寫入
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"Indexed {end}/{total}")

    print("=" * 60)
    print("Chroma indexing 完成")
    print("=" * 60)
    print(f"chunks 檔案: {CHUNKS_PATH}")
    print(f"向量庫路徑: {VECTORDB_PATH}")
    print(f"collection: {COLLECTION_NAME}")
    print(f"總筆數: {total}")


if __name__ == "__main__":
    main()