from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag.config import AppConfig
from vectorstores import ChromaRetriever, RetrievedChunk

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "eval" / "queries.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
DEFAULT_HIT_KS = [1, 3, 5]


def safe_filename_stem(path: Path) -> str:
    stem = path.stem.strip() or "queries"
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in stem)


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def load_queries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Eval queries not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        queries = json.load(f)

    if not isinstance(queries, list):
        raise ValueError("Eval queries must be a JSON list.")

    valid_queries: list[dict[str, Any]] = []
    for item in queries:
        if not isinstance(item, dict):
            continue

        reference = item.get("reference") or {}
        query = {
            "qid": str(item.get("qid", "")).strip(),
            "query": str(item.get("query", "")).strip(),
            "expected_doc_id": str(reference.get("doc_id", "")).strip(),
            "reference": reference,
        }
        if query["qid"] and query["query"] and query["expected_doc_id"]:
            valid_queries.append(query)

    if not valid_queries:
        raise ValueError("No valid eval queries with qid, query, and reference.doc_id found.")

    return valid_queries


def retrieved_doc_ids(chunks: list[RetrievedChunk]) -> list[str]:
    return [str(chunk.metadata.get("doc_id", "")).strip() for chunk in chunks]


def first_match_rank(doc_ids: list[str], expected_doc_id: str) -> int | None:
    for index, doc_id in enumerate(doc_ids, start=1):
        if doc_id == expected_doc_id:
            return index
    return None


def reciprocal_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / rank


def build_query_result(
    query_item: dict[str, Any],
    chunks: list[RetrievedChunk],
    hit_ks: list[int],
) -> dict[str, Any]:
    doc_ids = retrieved_doc_ids(chunks)
    expected_doc_id = query_item["expected_doc_id"]
    match_rank = first_match_rank(doc_ids, expected_doc_id)

    return {
        "qid": query_item["qid"],
        "query": query_item["query"],
        "expected_doc_id": expected_doc_id,
        "hit": match_rank is not None,
        "match_rank": match_rank,
        "reciprocal_rank": reciprocal_rank(match_rank),
        "hits_at_k": {f"hit@{k}": match_rank is not None and match_rank <= k for k in hit_ks},
        "retrieved": [
            {
                "rank": chunk.rank,
                "doc_id": str(chunk.metadata.get("doc_id", "")).strip(),
                "chunk_id": str(chunk.metadata.get("chunk_id", "")).strip(),
                "source": str(chunk.metadata.get("source", "")).strip(),
                "url": str(chunk.metadata.get("url", "")).strip(),
                "question": str(chunk.metadata.get("question", "")).strip(),
                "distance": chunk.distance,
            }
            for chunk in chunks
        ],
        "reference": query_item["reference"],
    }


def summarize_results(results: list[dict[str, Any]], hit_ks: list[int]) -> dict[str, Any]:
    total = len(results)
    summary: dict[str, Any] = {
        "total": total,
        "hit_rate": sum(1 for result in results if result["hit"]) / total,
        "mrr": sum(result["reciprocal_rank"] for result in results) / total,
    }

    for k in hit_ks:
        metric_name = f"hit@{k}"
        summary[metric_name] = (
            sum(1 for result in results if result["hits_at_k"][metric_name]) / total
        )

    return summary


def parse_hit_ks(value: str) -> list[int]:
    hit_ks = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not hit_ks or any(k <= 0 for k in hit_ks):
        raise argparse.ArgumentTypeError("hit-k values must be positive integers.")
    return hit_ks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Chroma retrieval quality.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help=f"Eval queries path. Default: {DEFAULT_QUERIES_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Default: eval/results/retrieval_eval_<timestamp>.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=max(DEFAULT_HIT_KS),
        help=f"Number of chunks to retrieve per query. Default: {max(DEFAULT_HIT_KS)}",
    )
    parser.add_argument(
        "--hit-ks",
        type=parse_hit_ks,
        default=DEFAULT_HIT_KS,
        help="Comma-separated hit@k values. Default: 1,3,5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")

    hit_ks = [k for k in args.hit_ks if k <= args.top_k]
    if not hit_ks:
        raise ValueError("At least one hit-k value must be less than or equal to --top-k.")

    config = AppConfig.from_env(PROJECT_ROOT)
    if not config.vectordb_path.exists():
        raise FileNotFoundError(
            f"Chroma database not found at {config.vectordb_path}. "
            "Run scripts/index_chroma.py first."
        )

    queries = load_queries(args.queries)
    retriever = ChromaRetriever(
        vectordb_path=config.vectordb_path,
        collection_name=config.collection_name,
        api_key=config.openai_api_key,
        embedding_model=config.embedding_model,
    )

    results = []
    for index, query_item in enumerate(queries, start=1):
        chunks = retriever.search(query_item["query"], top_k=args.top_k)
        result = build_query_result(query_item, chunks, hit_ks)
        results.append(result)

        status = "hit" if result["hit"] else "miss"
        print(
            f"[{index}/{len(queries)}] {query_item['qid']}: "
            f"{status}, rank={result['match_rank']}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_name = safe_filename_stem(args.queries)
    output_path = args.output or DEFAULT_RESULTS_DIR / f"retrieval_eval_{query_name}_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "queries_path": display_path(args.queries),
            "vectordb_path": display_path(config.vectordb_path),
            "collection_name": config.collection_name,
            "embedding_model": config.embedding_model,
            "top_k": args.top_k,
            "hit_ks": hit_ks,
        },
        "summary": summarize_results(results, hit_ks),
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    summary = payload["summary"]
    print()
    print("=" * 60)
    print("Retrieval eval complete")
    print("=" * 60)
    print(f"total: {summary['total']}")
    for k in hit_ks:
        print(f"hit@{k}: {summary[f'hit@{k}']:.3f}")
    print(f"hit_rate: {summary['hit_rate']:.3f}")
    print(f"mrr: {summary['mrr']:.3f}")
    print(f"output: {display_path(output_path)}")


if __name__ == "__main__":
    main()
