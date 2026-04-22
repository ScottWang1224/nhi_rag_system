import os
import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTORDB_PATH = PROJECT_ROOT / "data" / "vectordb" / "chroma"
QUERY_PATH = PROJECT_ROOT / "eval" / "queries.json"

COLLECTION_NAME = "nhi_rag_chunks"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 3


def print_result_block(rank, doc, meta, distance):
    print("=" * 80)
    print(f"Rank: {rank} | Distance: {distance}")
    print(f"doc_id: {meta.get('doc_id')}")
    print(f"question: {meta.get('question')}")
    print("-" * 80)
    print(doc)
    print()


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    client = chromadb.PersistentClient(path=str(VECTORDB_PATH))

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL,
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    with open(QUERY_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    for q in queries:
        query_text = q["query"]
        qid = q["qid"]

        print("\n" + "#" * 100)
        print(f"[QID] {qid}")
        print(f"[QUERY] {query_text}")
        print("#" * 100)

        results = collection.query(
            query_texts=[query_text],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            print_result_block(i, doc, meta, dist)


if __name__ == "__main__":
    main()