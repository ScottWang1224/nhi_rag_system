from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag import RAGService, build_service
from vectorstores import RetrievedChunk


def print_references(service_references) -> None:
    print("\nReferences")
    print("-" * 80)
    for reference in service_references:
        print(f"{reference.title}: {reference.url}")
    print("-" * 80)


def print_sources(chunks: list[RetrievedChunk]) -> None:
    print("\nSources")
    print("-" * 80)
    for chunk in chunks:
        metadata = chunk.metadata
        print(
            f"[{chunk.rank}] source={metadata.get('source')} "
            f"doc_id={metadata.get('doc_id')} "
            f"distance={chunk.distance}"
        )
        print(f"question: {metadata.get('question')}")
        print(f"url: {metadata.get('url')}")
        print(chunk.document)
        print("-" * 80)


def run_single_query(service: RAGService, query: str, *, top_k: int, show_sources: bool) -> None:
    result = service.answer_question(query, top_k=top_k)
    print("\nAnswer")
    print("=" * 80)
    print(result.answer)
    print("=" * 80)

    if show_sources:
        if result.retrieved_chunks:
            print_sources(result.retrieved_chunks)
        elif result.references:
            print_references(result.references)


def interactive_loop(service: RAGService, *, top_k: int, show_sources: bool) -> None:
    print("NHI RAG CLI")
    print("Type a question and press Enter. Use 'exit' or 'quit' to stop.")

    while True:
        try:
            query = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye")
            return

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Bye")
            return

        run_single_query(service, query, top_k=top_k, show_sources=show_sources)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NHI RAG CLI")
    parser.add_argument("--query", help="Run a single query and exit.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of retrieved chunks.")
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print the retrieved chunks after the answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, service = build_service(PROJECT_ROOT)
    top_k = args.top_k or config.top_k

    if args.query:
        run_single_query(
            service,
            args.query,
            top_k=top_k,
            show_sources=args.show_sources,
        )
        return

    interactive_loop(service, top_k=top_k, show_sources=args.show_sources)


if __name__ == "__main__":
    main()
