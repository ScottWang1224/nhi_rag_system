from playwright.sync_api import sync_playwright
import time
import json

url_list = [
    "https://www.nhi.gov.tw/ch/cp-3729-f2020-3246-1.html",
    "https://www.nhi.gov.tw/ch/cp-3784-d4ecb-3246-1.html",
    "https://www.nhi.gov.tw/ch/cp-3785-f169f-3246-1.html",
    "https://www.nhi.gov.tw/ch/cp-3786-d1076-3246-1.html",
    "https://www.nhi.gov.tw/ch/cp-3787-5fa86-3246-1.html",
    "https://www.nhi.gov.tw/ch/cp-3788-08cff-3246-1.html",
    "https://www.nhi.gov.tw/ch/cp-3793-e8fe7-3246-1.html",
    "https://www.nhi.gov.tw/ch/cp-3794-a3f84-3246-1.html"
]

results = []

def clean_text(text):
    return text.replace("\n", "").replace(" ", "").strip()

def print_title_article(url, idx):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url)

        title = page.locator("div.pageHeader h2").text_content()
        title = title.strip() if title else ""

        ps = page.locator("article.cpArticle p")
        content_list = []

        for i in range(ps.count()):
            text = ps.nth(i).text_content()
            if text:
                text = text.strip()
                if text:
                    content_list.append(text)

        if not content_list:
            lis = page.locator("article.cpArticle li")
            for i in range(lis.count()):
                text = lis.nth(i).text_content()
                if text:
                    text = text.strip()
                    if text:
                        content_list.append(text)

        content = "".join(content_list)
        content = clean_text(content)

        # 🔽 保留原本 print
        print(title)
        print("*"*60)
        print(content)
        print("*"*60)

        # 🔽 新增 JSON 收集
        results.append({
            "id": f"insurance_{idx+1:03d}",
            "question": title,
            "context": content,
            "source": "insurance",
            "url": url
        })

        browser.close()
        time.sleep(2)


for i in range(len(url_list)):
    print_title_article(url_list[i], i)

# 🔽 最後寫檔
with open("insurance.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)