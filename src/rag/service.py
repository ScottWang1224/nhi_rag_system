from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from vectorstores import ChromaRetriever, RetrievedChunk


@dataclass(slots=True)
class RAGAnswer:
    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]


class RAGService:
    def __init__(
        self,
        *,
        retriever: ChromaRetriever,
        api_key: str,
        answer_model: str,
        top_k: int,
    ) -> None:
        self._retriever = retriever
        self._client = OpenAI(api_key=api_key)
        self._answer_model = answer_model
        self._top_k = top_k

    def answer_question(self, query: str, *, top_k: int | None = None) -> RAGAnswer:
        final_top_k = top_k or self._top_k
        chunks = self._retriever.search(query, top_k=final_top_k)
        prompt = self._build_prompt(query, chunks)

        response = self._client.responses.create(
            model=self._answer_model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "你是台灣健保問答助理。請只根據提供的檢索內容回答。"
                                "如果資料不足，請明確說不知道或需要更多資料。"
                                "回答使用繁體中文，先給精簡答案，再補充重點。"
                                "引用來源時請用 [1]、[2] 這種格式。"
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
        )

        return RAGAnswer(
            query=query,
            answer=response.output_text.strip(),
            retrieved_chunks=chunks,
        )

    @staticmethod
    def _build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
        context_blocks: list[str] = []
        for chunk in chunks:
            metadata = chunk.metadata
            source = metadata.get("source", "unknown")
            question = metadata.get("question", "")
            url = metadata.get("url", "")
            context_blocks.append(
                "\n".join(
                    [
                        f"[{chunk.rank}] source: {source}",
                        f"[{chunk.rank}] question: {question}",
                        f"[{chunk.rank}] url: {url}",
                        f"[{chunk.rank}] content:",
                        chunk.document,
                    ]
                )
            )

        joined_context = "\n\n".join(context_blocks)
        return (
            f"使用者問題：{query}\n\n"
            "以下是可用檢索內容：\n"
            f"{joined_context}\n\n"
            "請根據以上內容回答，若答案可由多段資料支持，請整合後回答並附上引用。"
        )
