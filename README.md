# NHI RAG System

本專案是一個以台灣全民健康保險資料為主題的 RAG 問答系統，目標是處理不同型態的健保知識，並提供可實際互動的查詢介面。

目前系統包含：

- CLI 測試入口
- FastAPI 後端 API
- 簡易聊天式前端
- table / vector 雙路徑查詢流程

## 專案說明
健保資料同時包含敘述型文字與表格型資訊。若所有資料都以同一種方式處理，檢索效果通常不夠穩定。

因此本專案將資料分成兩類：

- 文字型資料：使用 chunk、embedding 與向量檢索
- 表格型資料：保留原本結構，使用獨立查詢流程

目前的系統設計是先判斷 query 的性質，再決定要查表還是走向量檢索。

## 系統流程
整體流程如下：

1. 使用者從 CLI 或前端送出 query
2. `src/rag/service.py` 作為核心流程控制層，接收並處理 query
3. 系統先進行基本安全與範圍判斷，例如：
   - 是否疑似 prompt injection
   - 是否為明顯非健保問題
4. `src/rag/router.py` 根據 query 特徵判斷路徑：
   - table route
   - vector route
5. 如果是 table route：
   - `src/tablestores/table_store.py` 查詢 `data/table/table.json`
6. 如果是 vector route：
   - `src/vectorstores/chroma_store.py` 從 Chroma 檢索相關 chunks
7. `src/rag/service.py` 根據不同路徑組裝 prompt，呼叫 LLM 生成答案
8. API 回傳 answer 與 references
9. 前端以聊天介面渲染結果

## Chunk 切分策略
### 文字型資料
文字型資料包含制度說明、QA 內容與一般敘述型知識，適合做 chunk 後再進行 embedding 與檢索。

利用 `scripts/analyze_lengths.py` 觀察原始資料的字數分布，再決定切分條件。

`analyze_lengths.py` 會先把每筆資料整理成：

- `question`
- `context`
- `question + context` 的完整文字

之後再統計每筆資料的字數長度，包含：

- 最小值
- 最大值
- 平均數
- 中位數
- P25 / P50 / P75 / P90 / P95

這樣做的目的，是先了解資料長度分布，再決定哪些資料適合保留成單一 chunk、哪些資料需要進一步切分。

在 `scripts/build_chunks.py` 中，目前切分條件如下：

- `SHORT_THRESHOLD = 80`
- `SINGLE_THRESHOLD = 200`
- `LONG_THRESHOLD = 350`
- `CHUNK_SIZE = 220`
- `CHUNK_OVERLAP = 40`

目前的策略不是把所有資料都切開，而是分層處理：

- 字數較短或中等的資料，直接保留成單一 chunk
- 字數超過 `LONG_THRESHOLD` 的資料，才交給 `RecursiveCharacterTextSplitter` 做進一步切分

這樣做的原因是：

- 短文本如果硬切，容易破壞原本完整語意
- 長文本如果完全不切，檢索時又可能帶入太多無關內容

實際切長文本時，系統會：

- 優先依照換行與標點切分
- 保留 `chunk_overlap`
- 清理切分後多餘的起始符號與空白
- 將每段答案重新包成 `問題 + 第 n 段答案` 的格式

這樣做的好處是，即使答案被拆成多段，retrieval 時仍然保留了問答語境，而不是只留下孤立的答案片段。

### 表格型資料
部分健保資料屬於結構化表格，例如：

- 保費分擔比例
- 補助規則
- 身分類別對應的數值資訊

這類資料若直接切成 chunk，常見問題包括：

- 欄位與數值關係被拆散
- 表格結構被破壞
- 向量相似度不一定適合查詢精確數值

因此本專案將這類資料獨立整理為：

- `data/table/table.json`

系統遇到表格型問題時，會直接查詢這份結構化資料，而不是走一般向量檢索。

## Table / Vector 分流設計
### Table route
table route 主要用於處理偏數值型、偏結構化的問題，例如：

- 比例是多少
- 誰負擔多少
- 政府補助幾%
- 某類身份對應什麼規則

### Vector route
vector route 主要處理偏說明型問題，例如：

- 名詞定義
- 資格條件
- 制度說明
- 辦理流程

### 目前作法
目前 router 主要根據：

- query 提示詞
- table match score
- row relevance

進行分流判斷。

## Prompt 與回答策略
回答生成邏輯目前集中在 `src/rag/service.py`。

prompt 設計原則為：

- 一律使用繁體中文回答
- 回答格式盡量維持一致：
  - 精簡答案
  - 補充重點
- table route 與 vector route 共用相同回答風格
- 根據資料來源不同，加入不同的資料使用限制

其中：

- table route 會更強調不得補充未提供的數值或比例
- vector route 會更強調只能根據檢索內容整理回答

## Prompt Safety 與範圍限制
目前系統已加入基礎版安全控制，內容包括：

- 攔截明顯疑似 prompt injection 的 query
- 拒答明顯與健保無關的問題
- 限制模型不得洩漏系統提示、內部規則或流程設計

這部分目前仍屬於第一版防護，後續仍可再強化，例如加入更完整的事件記錄與異常分析。

## API 與前端
### 後端
後端入口為：

- `api_main.py`

主要 API：

- `POST /api/chat`

回傳內容包含：

- `answer`
- `references`

### 前端
前端採用簡單聊天式介面，目前已實作：

- 繁體中文介面
- 使用者 / 助手對話泡泡
- 回答模擬串流顯示
- 參考資料超連結

## 主要檔案結構
- `main.py`：CLI 入口
- `api_main.py`：FastAPI 入口
- `src/rag/service.py`：核心流程控制與回答生成
- `src/rag/router.py`：table / vector 分流判斷
- `src/rag/bootstrap.py`：組裝 config、retriever、router、service
- `src/vectorstores/chroma_store.py`：Chroma 檢索層
- `src/tablestores/table_store.py`：表格資料查詢層
- `data/table/table.json`：表格型資料
- `scripts/index_chroma.py`：Chroma 建索引腳本

## 目前完成項目
目前已完成：

- 文字型資料的 chunking 與向量檢索流程
- Chroma 向量資料庫建索引與查詢
- RAG 問答流程
- CLI 測試入口
- FastAPI API
- 簡易聊天前端
- table / vector 分流
- 表格型資料的獨立查詢流程
- 基礎 prompt injection / out-of-scope 防護

## 目前限制
目前版本仍有幾個明確的限制：

- routing 仍以規則與簡單比對為主
- retrieval quality 尚未做完整 benchmark 驗證
- 尚未評估是否需要 rerank
- table query matching 仍可再加強欄位語意理解
- suspicious query 尚未正式做後端事件記錄

## 後續規劃
接下來預計優先處理：

- 建立 retrieval / answer evaluation 流程
- 驗證 top-k 與 chunk 策略
- 評估是否需要加入 reranker
- 強化 table query matching
- 補上異常 query 與疑似 injection query 的記錄機制

## 執行方式
### CLI
```powershell
uv run python main.py
```

單次查詢：

```powershell
uv run python main.py --query "什麼是健保給付?"
```

### FastAPI
```powershell
uv run uvicorn api_main:app --reload
```

啟動後開啟：

```text
http://127.0.0.1:8000
```