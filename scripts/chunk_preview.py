import json
from pathlib import Path
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ====================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "QA_dataset.json"  

SHORT_THRESHOLD = 80
SINGLE_THRESHOLD = 200
LONG_THRESHOLD = 350

SHORT_MERGE_GROUP_SIZE = 3

CHUNK_SIZE = 220
CHUNK_OVERLAP = 40
# ====================================


def build_text(item: dict) -> str:
    question = item.get("question", "").strip()
    context = item.get("context", "").strip()
    return f"問題：{question}\n答案：{context}"


def classify_length(text_len: int) -> str:
    if text_len < SHORT_THRESHOLD:
        return "short"
    elif text_len <= SINGLE_THRESHOLD:
        return "single"
    elif text_len <= LONG_THRESHOLD:
        return "medium"
    else:
        return "long"

def clean_chunk_text(text: str) -> str:
    text = text.strip()
    while text.startswith(("。", "，", "；", "\n", " ")):
        text = text[1:].lstrip()
    return text


def print_divider(title: str = "", width: int = 80) -> None:
    print("\n" + "=" * width)
    if title:
        print(title)
        print("=" * width)


def split_long_record(record: dict, splitter):
    question = record["question"].strip()
    answer = record["context"].strip()

    raw_chunks = splitter.split_text(answer)

    cleaned_chunks = []
    for chunk in raw_chunks:
        chunk = clean_chunk_text(chunk)
        if chunk:
            cleaned_chunks.append(chunk)

    total = len(cleaned_chunks)

    chunks = []
    for i, ans_chunk in enumerate(cleaned_chunks, start=1):
        chunk_text = f"問題：{question}\n答案（第{i}/{total}段）：{ans_chunk}"
        chunks.append(chunk_text)

    return chunks


def preview_long_chunks(records, splitter):
    long_records = [r for r in records if r["length_type"] == "long"]

    print_divider("LONG QA CHUNK PREVIEW")

    if not long_records:
        print("沒有 long 類型資料")
        return

    for record in long_records:
        print(f"\nID: {record['id']}")
        print(f"SOURCE: {record['source']}")
        print(f"CHAR_LEN: {record['char_len']}")
        print(f"QUESTION: {record['question']}")

        chunks = split_long_record(record, splitter)
        print(f"CHUNK_COUNT: {len(chunks)}")

        for i, chunk in enumerate(chunks, start=1):
            print(f"\n--- chunk {i} / {len(chunks)} | len={len(chunk)} ---")
            print(chunk)
            print("-" * 80)


def preview_short_merged(records: list[dict]) -> None:
    short_records = [r for r in records if r["length_type"] == "short"]

    print_divider("SHORT QA MERGE PREVIEW")

    if not short_records:
        print("沒有 short 類型資料")
        return

    grouped_by_source = defaultdict(list)
    for record in short_records:
        grouped_by_source[record["source"]].append(record)

    for source, items in grouped_by_source.items():
        print(f"\n[SOURCE] {source} | short_count={len(items)}")

        # 依長度排序，讓相近長度的先聚在一起
        items = sorted(items, key=lambda x: x["char_len"])

        merged_groups = [
            items[i:i + SHORT_MERGE_GROUP_SIZE]
            for i in range(0, len(items), SHORT_MERGE_GROUP_SIZE)
        ]

        for group_idx, group in enumerate(merged_groups, start=1):
            print(f"\n--- merged group {group_idx} | item_count={len(group)} ---")

            merged_text_parts = []
            for item_idx, item in enumerate(group, start=1):
                merged_text_parts.append(
                    f"[QA {item_idx}]\n"
                    f"問題：{item['question']}\n"
                    f"答案：{item['context']}"
                )

            merged_text = "\n\n".join(merged_text_parts)
            print(f"MERGED_LEN: {len(merged_text)}")
            print(merged_text)
            print("-" * 80)


def print_basic_stats(records: list[dict]) -> None:
    print_divider("LENGTH TYPE COUNTS")

    counts = defaultdict(int)
    source_counts = defaultdict(lambda: defaultdict(int))

    for r in records:
        counts[r["length_type"]] += 1
        source_counts[r["source"]][r["length_type"]] += 1

    print("整體分類統計：")
    for key in ["short", "single", "medium", "long"]:
        print(f"{key:>6}: {counts[key]}")

    print_divider("BY SOURCE")
    for source in sorted(source_counts.keys()):
        c = source_counts[source]
        print(
            f"{source:22} | "
            f"short={c['short']:2} | "
            f"single={c['single']:2} | "
            f"medium={c['medium']:2} | "
            f"long={c['long']:2}"
        )


def preview_medium_examples(records: list[dict], max_examples: int = 10) -> None:
    medium_records = [r for r in records if r["length_type"] == "medium"]

    print_divider("MEDIUM QA PREVIEW")

    if not medium_records:
        print("沒有 medium 類型資料")
        return

    for record in medium_records[:max_examples]:
        print(f"\nID: {record['id']}")
        print(f"SOURCE: {record['source']}")
        print(f"CHAR_LEN: {record['char_len']}")
        print(record["text"])
        print("-" * 80)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到檔案：{DATA_PATH.resolve()}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        text = build_text(item)
        char_len = len(text)

        records.append({
            "id": item.get("id", ""),
            "source": item.get("source", ""),
            "question": item.get("question", "").strip(),
            "context": item.get("context", "").strip(),
            "text": text,
            "char_len": char_len,
            "length_type": classify_length(char_len),
        })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", ""]
    )

    print_basic_stats(records)
    preview_medium_examples(records)
    preview_long_chunks(records, splitter)
    preview_short_merged(records)


if __name__ == "__main__":
    main()