"""
RAG Pipeline: Question -> Retrieval -> Generation -> Answer
"""
import os
import logging
from typing import Any, Dict, List
from retriever import retrieve_similar_chunks
from retriever import rerank_chunks
from db import connect
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
from wrapper import CustomE5Embeddings
import tiktoken

load_dotenv()
logger = logging.getLogger(__name__)
enc = tiktoken.get_encoding("cl100k_base")
EMBEDDING_MODEL = CustomE5Embeddings('intfloat/multilingual-e5-small')
SIMILARITY_THRESHOLD = 0.4

def build_prompt(context_chunks: List[Dict], question: str) -> str:
    """Buildin prompt using 9 instructions for generator"""
    context_text = "\n\n".join(
        f"[Документ: {chunk.get('document_id', 'Unknown')}, Страница: {chunk.get('page_number', '?')}]\n{chunk['text']}"
        for chunk in context_chunks
    )

    prompt = (
        "Ты — профессиональный финансовый аналитик. Твоя задача — отвечать на вопросы СТРОГО на основе предоставленного КОНТЕКСТА.\n\n"
        "ВАЖНО: Контекст предоставлен на английском языке. Вопрос задан на русском. Твой ответ должен быть на русском языке, но все факты и цифры бери ТОЛЬКО из английского контекста.\n\n"
        f"КОНТЕКСТ:\n{context_text}\n\n"
        f"ВОПРОС: {question}\n\n"
        "ПРАВИЛА ФОРМИРОВАНИЯ ОТВЕТА:\n"
        "1. Используй ТОЛЬКО информацию из раздела КОНТЕКСТ. Не используй свои внутренние знания.\n"
        "2. ПРИ ЦИТИРОВАНИИ ЦИФР ИЛИ ФАКТОВ ОБЯЗАТЕЛЬНО заключай их в кавычки и указывай источник, например: Выручка составила [85.8 млрд] [Источник: apple_10k, стр. 42].\n"
        "3. Если в контексте нет прямого ответа на вопрос, напиши ровно: 'В предоставленных документах нет информации для ответа на этот вопрос'.\n"
        "4. Каждое число или факт в твоем ответе должен быть подтвержден контекстом.\n"
        "5. В самом конце ответа обязательно укажи источник в формате: [Источник: document_id, страница X].\n"
        "6. Отвечай кратко, по делу, без вводных слов.\n"
        "7. Если в контексте недостаточно данных, чтобы покрыть все аспекты вопроса, явно перечисли, какие части не подтверждены документами, вместо того чтобы достраивать их.\n"
        "8. Цитату выдавай только если дан фактический ответ\n"
        "9. Отвечай исключительно на русском языке. Не воспроизводи текст из обучения. Если не можешь объединить факты из разных документов, перечисли их по отдельности с указанием источника для каждого\n\n"
        "ОТВЕТ:"
    )
    logger.info(f"PROMPT TOKENS: {len(enc.encode(prompt))}")

    return prompt


@retry(wait=wait_exponential(multiplier=1, min=1, max=4), stop=stop_after_attempt(3))
def call_ollama(prompt: str, model: str = "qwen2.5:7b") -> str:
    from ollama import Client
    """Call Ollama API and return generated text."""
    
    ollama_client = Client(timeout=10)
    response = ollama_client.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        options={"num_ctx": 8192}
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
                          database_url=os.getenv('DATABASE_URL'))) 