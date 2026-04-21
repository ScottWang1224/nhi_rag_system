import json
from pathlib import Path
from statistics import mean, median

DATA_PATH = Path("./QA_dataset.json")  

def build_text(item: dict) -> str:
    question = item.get("question", "").strip()
    context = item.get("context", "").strip()
    return f"問題：{question}\n答案：{context}"

def percentile(sorted_values, p: float):
    """簡單百分位數函式，p 例如 0.25 / 0.5 / 0.9"""
    if not sorted_values:
        return 0
    idx = int((len(sorted_values) - 1) * p)
    return sorted_values[idx]

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

records = []
for item in data:
    text = build_text(item)
    records.append({
        "id": item.get("id", ""),
        "source": item.get("source", ""),
        "question": item.get("question", ""),
        "text": text,
        "char_len": len(text),
        "question_len": len(item.get("question", "").strip()),
        "context_len": len(item.get("context", "").strip()),
    })

lengths = sorted(r["char_len"] for r in records)

print("=" * 60)
print("整體統計")
print("=" * 60)
print(f"總筆數: {len(records)}")
print(f"最短: {min(lengths)}")
print(f"最長: {max(lengths)}")
print(f"平均: {mean(lengths):.2f}")
print(f"中位數: {median(lengths):.2f}")
print(f"P25: {percentile(lengths, 0.25)}")
print(f"P50: {percentile(lengths, 0.50)}")
print(f"P75: {percentile(lengths, 0.75)}")
print(f"P90: {percentile(lengths, 0.90)}")
print(f"P95: {percentile(lengths, 0.95)}")

print("\n" + "=" * 60)
print("最短 10 筆")
print("=" * 60)
for r in sorted(records, key=lambda x: x["char_len"])[:10]:
    print(f"{r['id']:15} | {r['source']:22} | len={r['char_len']:4} | {r['question']}")

print("\n" + "=" * 60)
print("最長 10 筆")
print("=" * 60)
for r in sorted(records, key=lambda x: x["char_len"], reverse=True)[:10]:
    print(f"{r['id']:15} | {r['source']:22} | len={r['char_len']:4} | {r['question']}")

# 依 source 分組觀察
source_stats = {}
for r in records:
    source = r["source"]
    source_stats.setdefault(source, []).append(r["char_len"])

print("\n" + "=" * 60)
print("各 source 分布")
print("=" * 60)
for source, vals in sorted(source_stats.items()):
    vals_sorted = sorted(vals)
    print(
        f"{source:22} | n={len(vals):3} | "
        f"min={min(vals_sorted):4} | "
        f"median={median(vals_sorted):6.1f} | "
        f"mean={mean(vals_sorted):6.1f} | "
        f"p90={percentile(vals_sorted, 0.90):4} | "
        f"max={max(vals_sorted):4}"
    )