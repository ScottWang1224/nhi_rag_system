from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
import time
import json

base_url = "https://www.nhi.gov.tw"
url_list = []
results = []

def clean_text(text):
    return text.replace("\n", "").replace(" ", "").strip()

def print_title_article(url,idx):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url)
        title = page.locator("div.pageHeader h2").text_content()
        title = title.strip() if title else ""

        # 先抓 p
        ps = page.locator("article.cpArticle p")
        content_list = []
        for i in range(ps.count()):
            text = ps.nth(i).text_content()
            if text:
                text = text.strip()
                if text:
                    content_list.append(text)
        # 如果 p 沒抓到內容，再抓 li
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
        print(title)
        print("*"*40)
        print(content)
        print("*"*40)
        results.append({
            "id": f"other_{idx+1:03d}",
            "question": title,
            "context": content,
            "source": "other",
            "url": url
        })
        browser.close()
    time.sleep(2)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://www.nhi.gov.tw/ch/lp-3249-1.html?pi=1&ps=40")

    links = page.locator("section.list li a")
    count = links.count()

    print("找到連結數量:", count)

    for i in range(count):
        a = links.nth(i)
        text = a.text_content()
        href = a.get_attribute("href")

        text = text.strip() if text else ""
        full_url = urljoin(base_url, href) if href else ""
        url_list.append(full_url)

    browser.close()

time.sleep(2)

for i in range(count):
    url = url_list[i]
    print_title_article(url,i)

with open("other.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

