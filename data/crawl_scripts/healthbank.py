import json
from playwright.sync_api import sync_playwright

results = []

url = "https://myhealthbank.nhi.gov.tw/IHKE3000/IHKE3115S03"

def clean_text(text):
    return text.replace("\n", "").replace(" ", "").strip()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_timeout(2000)

    buttons = page.locator("div.accordion-header button")
    bodies = page.locator("div.accordion-body.edit")

    count = min(buttons.count(), bodies.count())

    for i in range(count):
        button = buttons.nth(i)
        body = bodies.nth(i)

        title = button.get_attribute("title")
        if not title:
            title = button.inner_text().strip()

        print(f"\n【問題 {i+1}】{title}")
        print("-" * 60)

        content_list = []

        lis = body.locator("li")
        if lis.count() > 0:
            for j in range(lis.count()):
                text = lis.nth(j).inner_text().strip()
                content_list.append(text)
                print(f"  - {text}")
        else:
            text = body.inner_text().strip()
            content_list.append(text)
            print(text)

        content = "".join(content_list)
        content = clean_text(content)

        # 🔽 收集
        results.append({
            "id": f"healthbank_{i+1:03d}",
            "question": title,
            "context": content,
            "source": "healthbank",
            "url": url
        })

    browser.close()

with open("healthbank.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)