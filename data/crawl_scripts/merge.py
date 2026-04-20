import json

def load_json(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

data = []
data += load_json("insurance.json")
data += load_json("healthbank.json")
data += load_json("other.json")
data += load_json("pdf.json")

with open("../QA_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("完成合併")