"""
RAG Pipeline: Question -> Retrieval -> Generation -> Answer
"""
import os
import logging
from typing import Any, Dict, List
from retriever import retrieve_similar_chunks
from retriever import rerank_chunks
from db import connect
from config import RERANKER_MODEL_NAME, DATABASE_URL, NUM_CTX, GENERATOR_MODEL, EMBEDDING_MODEL_NAME
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
from sentence_transformers import SentenceTransformer
import tiktoken
import re
load_dotenv()
logger = logging.getLogger(__name__)
enc = tiktoken.get_encoding("cl100k_base")
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
SIMILARITY_THRESHOLD = 0.4

REFUSAL_RE = re.compile(
    r"(нет информации|не указано|не содержится|не приводится|"
    r"not enough information|insufficient information|no information)",
    re.IGNORECASE,
)

SOURCE_RE = re.compile(
    r"\s*\[?\s*(Источник|Source)[^\]]*\]?\s*\.?",
    re.IGNORECASE,
)

NUMBER_RE = re.compile(r"\d")

CANONICAL_REFUSAL = (
    "В предоставленных документах нет информации для ответа на этот вопрос."
)

FRAGMENT_SOURCE_RE = re.compile(
    r"\[?\s*(?:Источник|Source)\s*:\s*Фрагмент\s+(\d+)\s*\]?",
    re.IGNORECASE,
)


def resolve_fragment_sources(answer: str, contexts: List[Dict[str, Any]]) -> str:
    """Заменяет [Источник: Фрагмент N] на реальные document_id/page_number.

    Контекст приходит уже отсортированным по rerank_score из retrieve_context,
    поэтому индекс N в промпте соответствует позиции chunk в этом же списке.
    """
    def replace(match: re.Match) -> str:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(contexts):
            chunk = contexts[idx]
            doc_id = chunk.get("document_id") or "?"
            page = chunk.get("page_number") or "?"
            return f"[Источник: {doc_id}, страница {page}]"
        # Если модель выдумала номер вне диапазона — оставляем как есть,
        # чтобы sanitize_answer мог его поймать/убрать при отказе.
        return match.group(0)

    return FRAGMENT_SOURCE_RE.sub(replace, answer)


def generate_rag_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return CANONICAL_REFUSAL

    prompt = build_prompt(contexts, question)
    raw_answer = call_ollama(prompt)
    resolved_answer = resolve_fragment_sources(raw_answer, contexts)
    return sanitize_answer(resolved_answer)

def sanitize_answer(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return CANONICAL_REFUSAL

    # 1. Сначала снимаем источники, потом смотрим на числа.
    stripped = SOURCE_RE.sub("", text).strip()

    # 2. Если после снятия источника остался чистый отказ без чисел — канонизируем.
    if REFUSAL_RE.search(stripped) and not NUMBER_RE.search(stripped):
        return CANONICAL_REFUSAL

    # 3. Иначе режем по предложениям и вырезаем источник только из отказных
    #    предложений, где в теле нет чисел.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for sentence in sentences:
        body = SOURCE_RE.sub("", sentence).strip()
        if REFUSAL_RE.search(body) and not NUMBER_RE.search(body):
            cleaned.append(CANONICAL_REFUSAL)
        else:
            cleaned.append(sentence.strip())
    return " ".join(cleaned).strip()

def build_prompt(context_chunks: List[Dict[str, Any]], question: str) -> str:
    parts = []

    for i, chunk in enumerate(context_chunks, start=1):
        doc_id = chunk.get("document_id") or "?"
        page = chunk.get("page_number") or "?"
        section = chunk.get("section") or "без секции"
        kind = "таблица" if chunk.get("is_table") else "текст"

        parts.append(
            f"[Фрагмент {i}: {doc_id}, страница {page}, секция: {section}, тип: {kind}]\n"
            f"{chunk.get('text', '').strip()}"
        )

    context = "\n\n".join(parts)

    return f"""Ты — финансовый ассистент. Ответь на вопрос только на основе предоставленных фрагментов.

Правила:
1. Отвечай кратко, по существу, на русском языке.
2. Если фрагменты содержат ответ, дай фактический ответ.
3. Если отвечаешь фактом, укажи источник только в формате:
   [Источник: Фрагмент N]
   где N — номер фрагмента из списка выше.
4. НЕ указывай document_id, название компании или номер страницы напрямую.
5. Если в фрагментах нет информации для ответа, напиши ровно:
   "В предоставленных документах нет информации для ответа на этот вопрос."
   В этом случае НЕ указывай источник, НЕ объясняй причину, НЕ добавляй предположения.
6. Числа, валюты, проценты и единицы измерения переноси ровно так, как они записаны во фрагментах.
   Не добавляй слова "млн", "миллионов", "миллиард", "долларов", "евро" и символы "$", "€", "£",
   если они не стоят непосредственно рядом с числом во фрагменте.
7. Не додумывай отсутствующие данные. Не объединяй числа из разных фрагментов в новое утверждение,
   если вопрос прямо не просит сравнение.
8. Не воспроизводи посторонний текст из обучения. Не пиши на китайском или английском, если вопрос на русском.

Примеры:

Фрагмент 1: Total net sales 383,285
Вопрос: Какова общая чистая выручка Apple?
Ответ: 383,285 [Источник: Фрагмент 1]

Фрагмент 1: Apple Vision Pro expected to be available in early 2024
Вопрос: Какую долю выручки Apple составил сегмент Apple Vision Pro?
Ответ: В предоставленных документах нет информации для ответа на этот вопрос.

Фрагмент 1: imposed a €2.4 billion fine
Вопрос: Какой штраф наложила Европейская комиссия?
Ответ: €2.4 billion [Источник: Фрагмент 1]

Фрагменты:
{context}

Вопрос: {question}
Ответ:
"""


@retry(wait=wait_exponential(multiplier=1, min=1, max=4), stop=stop_after_attempt(3))
def call_ollama(prompt: str, model: str = GENERATOR_MODEL) -> str:
    from ollama import Client

    client = Client(timeout=120)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_ctx": 8192,
            "temperature": 0.0,
        },
    )
    return response.message.content

def answer_question(
    question: str, 
    database_url: str, 
    top_k: int,
    model_name: str = "qwen2.5:7b",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """End-to-end RAG: retrieve chunks, build prompt, generate answer."""
    with connect(database_url) as conn:
        similar_chunks = retrieve_similar_chunks(
                                                connection=conn,
                                                top_k=top_k, 
                                                query_text=question, 
                                                model=EMBEDDING_MODEL)
    best_chunks = rerank_chunks(question, similar_chunks, top_k=5)
    if not best_chunks or best_chunks[0]['similarity'] <= similarity_threshold:
        return {'answer': 'No answer in documents', 'sources': []}
    prompt = build_prompt(best_chunks, question)
    response = call_ollama(prompt, model=model_name)

    return {'answer': response + '\n', 'sources': best_chunks}

if __name__ == "__main__":
    while True:
        question = input('Enter your ptompt:\n')
        print(answer_question(question=question, 
                          database_url=DATABASE_URL))