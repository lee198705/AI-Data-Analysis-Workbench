from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jieba
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "统计理论知识语料库.xlsx"
STOPWORDS_PATH = PROJECT_ROOT / "ciyun.txt"


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    source: str
    chunk_id: int = 0


@dataclass(frozen=True)
class RetrievalResult:
    chunk: DocumentChunk
    score: float


@lru_cache(maxsize=1)
def load_stopwords() -> frozenset[str]:
    if not STOPWORDS_PATH.exists():
        return frozenset()
    return frozenset(STOPWORDS_PATH.read_text(encoding="utf-8").split())


def normalize_text(text: object) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return cleaned


def tokenize(text: str) -> list[str]:
    stopwords = load_stopwords()
    tokens = []
    for token in jieba.lcut(normalize_text(text).lower()):
        token = token.strip()
        if not token or token in stopwords:
            continue
        if re.fullmatch(r"[\W_]+", token):
            continue
        tokens.append(token)
    return tokens


def chunks_from_text(
    text: str,
    source: str,
    chunk_size: int = 520,
    overlap: int = 80,
) -> list[DocumentChunk]:
    text = normalize_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    chunk_id = 1
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(
            DocumentChunk(
                content=text[start:end],
                source=source,
                chunk_id=chunk_id,
            )
        )
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
        chunk_id += 1
    return chunks


def chunks_from_dataframe(df: pd.DataFrame, source: str, max_rows: int = 200) -> list[DocumentChunk]:
    if df.empty:
        return []

    preview = df.head(max_rows).copy()
    content = preview.to_csv(index=False)
    summary = (
        f"表格来源：{source}\n"
        f"总行数：{len(df)}，总列数：{len(df.columns)}\n"
        f"字段：{', '.join(map(str, df.columns))}\n\n"
        f"前 {len(preview)} 行数据：\n{content}"
    )
    return chunks_from_text(summary, source=source, chunk_size=900, overlap=120)


@lru_cache(maxsize=1)
def load_knowledge_chunks() -> tuple[DocumentChunk, ...]:
    if not KNOWLEDGE_BASE_PATH.exists():
        return tuple()

    df = pd.read_excel(KNOWLEDGE_BASE_PATH)
    if df.empty:
        return tuple()

    title_col = "标题" if "标题" in df.columns else df.columns[0]
    content_col = "内容" if "内容" in df.columns else df.columns[min(1, len(df.columns) - 1)]
    chunks: list[DocumentChunk] = []
    for row_index, row in df.iterrows():
        title = normalize_text(row.get(title_col, ""))
        content = normalize_text(row.get(content_col, ""))
        if not title and not content:
            continue
        source = f"统计理论知识语料库：{title or f'第 {row_index + 1} 条'}"
        text = f"标题：{title}\n内容：{content}"
        chunks.extend(chunks_from_text(text, source=source, chunk_size=640, overlap=90))

    return tuple(chunks)


class BM25Retriever:
    def __init__(self, chunks: list[DocumentChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(chunk.content) for chunk in chunks]
        self.doc_freqs = [Counter(doc) for doc in self.documents]
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        document_count = len(self.documents)
        term_document_counts: Counter[str] = Counter()
        for doc in self.documents:
            term_document_counts.update(set(doc))

        return {
            term: math.log(1 + (document_count - freq + 0.5) / (freq + 0.5))
            for term, freq in term_document_counts.items()
        }

    def search(self, query: str, top_k: int = 5, min_score: float = 0.05) -> list[RetrievalResult]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        scored: list[RetrievalResult] = []
        for index, freqs in enumerate(self.doc_freqs):
            score = self._score_document(query_tokens, freqs, self.doc_lengths[index])
            if score >= min_score:
                scored.append(RetrievalResult(chunk=self.chunks[index], score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _score_document(self, query_tokens: list[str], freqs: Counter[str], doc_length: int) -> float:
        score = 0.0
        for term in query_tokens:
            term_freq = freqs.get(term, 0)
            if term_freq == 0:
                continue
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1)
            )
            score += self.idf.get(term, 0.0) * (term_freq * (self.k1 + 1)) / denominator
        return score


def retrieve_context(
    question: str,
    uploaded_chunks: list[DocumentChunk] | None = None,
    top_k: int = 5,
) -> list[RetrievalResult]:
    knowledge_results = BM25Retriever(list(load_knowledge_chunks())).search(question, top_k=top_k)

    if not uploaded_chunks:
        return knowledge_results[:top_k]

    uploaded_results = BM25Retriever(uploaded_chunks).search(question, top_k=top_k, min_score=0.01)
    upload_keywords = ["上传", "文件", "文档", "资料", "表格", "数据集", "字段", "这份", "这个"]
    is_upload_question = any(keyword in question for keyword in upload_keywords)
    upload_weight = 2.0 if is_upload_question else 1.2
    upload_bonus = 6.0 if is_upload_question else 0.0
    boosted_uploads = [
        RetrievalResult(chunk=result.chunk, score=result.score * upload_weight + upload_bonus)
        for result in uploaded_results
    ]

    merged = boosted_uploads + knowledge_results
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]


def build_rag_prompt(question: str, results: list[RetrievalResult]) -> str:
    if not results:
        return (
            "你是一个统计学问答助手。当前没有检索到可靠参考资料。"
            "请直接回答用户问题；如果信息不足，请明确说明。\n\n"
            f"用户问题：{question}"
        )

    context_blocks = []
    for index, result in enumerate(results, start=1):
        context_blocks.append(
            f"[来源{index}] {result.chunk.source}\n{result.chunk.content}"
        )
    context = "\n\n".join(context_blocks)
    return (
        "你是一个统计学问答助手。请优先依据下面的参考资料回答，"
        "不要编造资料中没有的事实。回答要清晰、简洁，并在关键结论后标注来源编号，"
        "例如“均值用于描述一般水平[来源1]”。如果资料不足，请说明不足之处。\n\n"
        f"参考资料：\n{context}\n\n"
        f"用户问题：{question}"
    )
