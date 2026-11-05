"""
RAG Pipeline: Question -> Retrieval -> Generation -> Answer
"""
import os
import logging
from typing import Any, Dict, List
from retriever import retrieve_similar_chunks
from db import connect
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
SIMILARITY_THRESHOLD = 0.4

def build_prompt(context: List[Dict[str, Any]], question: str) -> str:
    """Build structured prompt for financial QA."""

    instructions = (
        "Ты - профессиональный финансовый аналитик. Отвечай на русском языке.\n"
        "Если в документах нет ответа на вопрос, не выдумывай и скажи об этом.\n"
        "Если ответ есть, укажи название документа и страницу.\n"
        f"Вопрос: {question}\n"
        "Контекст:\n"
    )

    for chunk in context:
        instructions += (
            f"Document: {chunk['document_id']}, Page: {chunk['page_number']}\n"
            f"Text: {chunk['text']}\n\n"
        )
    return instructions


@retry(wait=wait_exponential(multiplier=1, min=1, max=4), stop=stop_after_attempt(3))
def call_ollama(prompt: str, model: str = "qwen2.5:7b") -> str:
    from ollama import Client
    """Call Ollama API and return generated text."""
    
    ollama_client = Client(timeout=10)
    response = ollama_client.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response.message.content

def answer_question(
    question: str, 
    database_url: str, 
    model_name: str = "qwen2.5:7b",
    top_k: int = 5,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """End-to-end RAG: retrieve chunks, build prompt, generate answer."""
    with connect(database_url) as conn:
        similar_chunks = retrieve_similar_chunks(
                                                connection=conn,
                                                top_k=top_k, 
                                                query_text=question, 
                                                model=EMBEDDING_MODEL)
    if not similar_chunks or similar_chunks[0]['similarity'] <= similarity_threshold:
        return {'answer': 'No answer in documents', 'sources': []}
    prompt = build_prompt(similar_chunks, question)
    response = call_ollama(prompt, model=model_name)

    return {'answer': response + '\n', 'sources': similar_chunks}

if __name__ == "__main__":
    while True:
        question = input('Enter your ptompt:\n')
        print(answer_question(question=question, 
                          database_url=os.getenv('DATABASE_URL')))