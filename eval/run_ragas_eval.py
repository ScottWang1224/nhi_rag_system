from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag.bootstrap import build_service

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "eval" / "queries_realistic.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


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
            "ground_truth": str(item.get("ground_truth", "")).strip(),
            "reference": reference,
            "expected_doc_id": str(reference.get("doc_id", "")).strip(),
            "original_query": str(item.get("original_query", "")).strip(),
            "query_type": str(item.get("query_type", "")).strip(),
        }
        if query["qid"] and query["query"] and query["ground_truth"]:
            valid_queries.append(query)

    if not valid_queries:
        raise ValueError("No valid eval queries with qid, query, and ground_truth found.")

    return valid_queries


def parse_metrics(value: str) -> list[str]:
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    if not metrics:
        raise argparse.ArgumentTypeError("At least one metric is required.")
    return metrics


def load_ragas_metric_objects(metric_names: list[str]) -> list[Any]:
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    available_metrics = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }
    unknown_metrics = [name for name in metric_names if name not in available_metrics]
    if unknown_metrics:
        raise ValueError(
            "Unsupported RAGAS metrics: "
            + ", ".join(unknown_metrics)
            + ". Supported metrics: "
            + ", ".join(sorted(available_metrics))
        )

    return [available_metrics[name] for name in metric_names]


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    return value


def mean_score(values: list[Any]) -> float | None:
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, int | float) and not math.isnan(float(value))
    ]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def table_context_records(service: Any, query: str) -> tuple[list[str], list[dict[str, Any]]]:
    decision = service._router.route(query)
    table_matches = decision.table_matches
    table_contexts = [service._table_store.format_context(match) for match in table_matches]
    retrieved_tables = [
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
    return table_contexts, retrieved_tables


def build_generation_record(query_item: dict[str, Any], rag_answer: Any, service: Any) -> dict[str, Any]:
    retrieved_chunks = rag_answer.retrieved_chunks
    contexts = [chunk.document for chunk in retrieved_chunks]
    retrieved = [
        {
            "rank": chunk.rank,
            "doc_id": str(chunk.metadata.get("doc_id", "")).strip(),
            "chunk_id": str(chunk.metadata.get("chunk_id", "")).strip(),
            "source": str(chunk.metadata.get("source", "")).strip(),
            "url": str(chunk.metadata.get("url", "")).strip(),
            "question": str(chunk.metadata.get("question", "")).strip(),
            "distance": chunk.distance,
        }
        for chunk in retrieved_chunks
    ]
    table_contexts: list[str] = []
    retrieved_tables: list[dict[str, Any]] = []
    if rag_answer.route_mode == "table":
        table_contexts, retrieved_tables = table_context_records(service, query_item["query"])
        contexts = table_contexts + contexts

    return {
        "qid": query_item["qid"],
        "query": query_item["query"],
        "original_query": query_item["original_query"],
        "query_type": query_item["query_type"],
        "answer": rag_answer.answer,
        "ground_truth": query_item["ground_truth"],
        "contexts": contexts,
        "retrieved": retrieved,
        "retrieved_tables": retrieved_tables,
        "route_mode": rag_answer.route_mode,
        "reference": query_item["reference"],
    }


def run_ragas(generation_records: list[dict[str, Any]], metric_names: list[str], batch_size: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate

    samples = [
        SingleTurnSample(
            user_input=record["query"],
            response=record["answer"],
            retrieved_contexts=record["contexts"],
            reference=record["ground_truth"],
        )
        for record in generation_records
    ]
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset,
        metrics=load_ragas_metric_objects(metric_names),
        batch_size=batch_size,
        raise_exceptions=False,
    )

    scores = sanitize_for_json(result.scores)
    summary = {
        metric_name: mean_score([score.get(metric_name) for score in scores])
        for metric_name in metric_names
    }
    return sanitize_for_json(summary), scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAG answers and evaluate them with RAGAS.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help=f"Eval queries path. Default: {DEFAULT_QUERIES_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Default: eval/results/ragas_eval_<queries>_<timestamp>.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Number of chunks to retrieve per query. Default: RAG_TOP_K from .env/config.",
    )
    parser.add_argument(
        "--metrics",
        type=parse_metrics,
        default=DEFAULT_METRICS,
        help="Comma-separated RAGAS metrics. Default: " + ",".join(DEFAULT_METRICS),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="RAGAS evaluation batch size. Default: no batching.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate RAG answers and contexts; skip RAGAS scoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    queries = load_queries(args.queries)
    config, service = build_service(PROJECT_ROOT)
    final_top_k = args.top_k or config.top_k

    generation_records = []
    for index, query_item in enumerate(queries, start=1):
        rag_answer = service.answer_question(query_item["query"], top_k=final_top_k)
        record = build_generation_record(query_item, rag_answer, service)
        generation_records.append(record)
        print(
            f"[{index}/{len(queries)}] {query_item['qid']}: "
            f"route={record['route_mode']}, contexts={len(record['contexts'])}"
        )

    ragas_summary: dict[str, Any] = {}
    ragas_scores: list[dict[str, Any]] = []
    if not args.generate_only:
        print()
        print("Running RAGAS scoring...")
        ragas_summary, ragas_scores = run_ragas(
            generation_records,
            metric_names=args.metrics,
            batch_size=args.batch_size,
        )

    results = []
    for record, scores in zip(generation_records, ragas_scores or [{} for _ in generation_records]):
        results.append({**record, "ragas_scores": scores})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_name = safe_filename_stem(args.queries)
    output_path = args.output or DEFAULT_RESULTS_DIR / f"ragas_eval_{query_name}_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "queries_path": display_path(args.queries),
            "vectordb_path": display_path(config.vectordb_path),
            "collection_name": config.collection_name,
            "embedding_model": config.embedding_model,
            "answer_model": config.answer_model,
            "top_k": final_top_k,
            "metrics": [] if args.generate_only else args.metrics,
            "generate_only": args.generate_only,
        },
        "summary": {
            "total": len(results),
            "ragas": ragas_summary,
        },
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")

    print()
    print("=" * 60)
    print("RAG eval complete")
    print("=" * 60)
    print(f"total: {len(results)}")
    if ragas_summary:
        for metric_name, score in ragas_summary.items():
            score_text = "nan" if score is None else f"{score:.3f}"
            print(f"{metric_name}: {score_text}")
    else:
        print("ragas: skipped")
    print(f"output: {display_path(output_path)}")


if __name__ == "__main__":
    main()
