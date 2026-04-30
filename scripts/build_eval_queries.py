from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw_data" / "QA_dataset.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "queries.json"

DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SEED = 42


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input dataset must be a JSON list.")

    records: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        record = {
            "id": str(item.get("id", "")).strip(),
            "question": str(item.get("question", "")).strip(),
            "context": str(item.get("context", "")).strip(),
            "source": str(item.get("source", "")).strip(),
            "url": str(item.get("url", "")).strip(),
        }
        if record["id"] and record["question"] and record["context"]:
            records.append(record)

    if not records:
        raise ValueError("No valid records found in input dataset.")

    return records


def filter_by_sources(
    records: list[dict[str, Any]],
    sources: list[str] | None,
) -> list[dict[str, Any]]:
    if not sources:
        return records

    wanted = {source.strip() for source in sources if source.strip()}
    return [record for record in records if record["source"] in wanted]


def sample_records(
    records: list[dict[str, Any]],
    *,
    sample_size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size is None or sample_size >= len(records):
        return list(records)

    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0.")

    rng = random.Random(seed)
    return rng.sample(records, sample_size)


def build_eval_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "qid": record["id"],
            "query": record["question"],
            "ground_truth": record["context"],
            "reference": {
                "doc_id": record["id"],
                "source": record["source"],
                "url": record["url"],
            },
        }
        for record in records
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build eval/queries.json by sampling raw QA records.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Raw QA dataset path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output eval queries path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of records to sample. Default: {DEFAULT_SAMPLE_SIZE}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all valid records instead of sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible sampling. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Limit sampling to a source. Can be used multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = load_records(args.input)
    filtered_records = filter_by_sources(records, args.source)
    if not filtered_records:
        raise ValueError("No records matched the selected source filters.")

    sample_size = None if args.all else args.sample_size
    sampled_records = sample_records(
        filtered_records,
        sample_size=sample_size,
        seed=args.seed,
    )
    sampled_records.sort(key=lambda record: record["id"])

    eval_items = build_eval_items(sampled_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(eval_items, f, ensure_ascii=False, indent=2)
        f.write("\n")

    source_counts: dict[str, int] = {}
    for item in eval_items:
        source = item["reference"]["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"Wrote {len(eval_items)} eval queries to {args.output}")
    print(f"Seed: {args.seed}")
    print(f"Sources: {source_counts}")


if __name__ == "__main__":
    main()
