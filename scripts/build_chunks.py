import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========= 路徑與參數 =========
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "QA_dataset.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"

SHORT_THRESHOLD = 80
SINGLE_THRESHOLD = 200
LONG_THRESHOLD = 350

CHUNK_SIZE = 220
CHUNK_OVERLAP = 40
# =============================


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


def split_long_record(record: dict, splitter: RecursiveCharacterTextSplitter) -> list[str]:
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


def build_single_chunk(record: dict) -> dict:
    content = build_text(record)
    return {
        "chunk_id": f"{record['id']}_chunk_1",
        "doc_id": record["id"],
        "source": record["source"],
        "url": record["url"],
        "question": record["question"],
        "chunk_type": "single",
        "length_type": record["length_type"],
        "chunk_index": 1,
        "chunk_total": 1,
        "content": content,
        "char_len": len(content),
    }


def build_long_chunks(record: dict, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    split_chunks = split_long_record(record, splitter)
    total = len(split_chunks)

    results = []
    for i, chunk_text in enumerate(split_chunks, start=1):
        results.append({
            "chunk_id": f"{record['id']}_chunk_{i}",
            "doc_id": record["id"],
            "source": record["source"],
            "url": record["url"],
            "question": record["question"],
            "chunk_type": "long_split",
            "length_type": record["length_type"],
            "chunk_index": i,
            "chunk_total": total,
            "content": chunk_text,
            "char_len": len(chunk_text),
        })
    return results


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到檔案：{DATA_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        record = {
            "id": item.get("id", "").strip(),
            "source": item.get("source", "").strip(),
            "url": item.get("url", "").strip(),
            "question": item.get("question", "").strip(),
            "context": item.get("context", "").strip(),
        }

        text = build_text(record)
        record["text"] = text
        record["char_len"] = len(text)
        record["length_type"] = classify_length(record["char_len"])
        records.append(record)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", ""]
    )

    all_chunks = []

    for record in records:
        if record["length_type"] == "long":
            all_chunks.extend(build_long_chunks(record, splitter))
        else:
            all_chunks.append(build_single_chunk(record))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("chunks 建立完成")
    print("=" * 60)
    print(f"原始文件數: {len(records)}")
    print(f"輸出 chunk 數: {len(all_chunks)}")

    single_count = sum(1 for c in all_chunks if c["chunk_type"] == "single")
    long_split_count = sum(1 for c in all_chunks if c["chunk_type"] == "long_split")

    print(f"single chunks: {single_count}")
    print(f"long_split chunks: {long_split_count}")


if __name__ == "__main__":
    main()