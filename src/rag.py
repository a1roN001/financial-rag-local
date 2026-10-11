"""
RAG Pipeline: Question -> Retrieval -> Generation -> Answer
"""
import os
import logging
from typing import Dict, List, Any
from retriever import retrieve_similar_chunks
from db import connect
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def build_prompt(context: str, question: str) -> str:
    """Build structured prompt for financial QA."""

    instructions = 'Ты - профессиональный финансовый аналитик, отвечай соотвествующе" \
    "Если в представленных документах нет ответа на заданный вопрос, не выдумывай, так и скажи что ответа нет" \
    "Если найдешь ответ в предоставленном контексте, выведи страницу документа на которой он есть" \
    "Вот контекст твоего ответа: '
    for dic in context:
        instructions += f"Document: {dic['document_id']}, Page: {dic['page_number']}, text: {dic["text"]}"
    return instructions

def call_ollama(prompt: str, model: str = "qwen2.5:7b") -> str:
    from ollama import chat
    from ollama import ChatResponse
    """Call Ollama API and return generated text."""

    response : ChatResponse = chat(model=model, messages=[{'role': 'user', 'content': prompt}])    
    return response

def answer_question(
    question: str, 
    database_url: str, 
    model_name: str = "qwen2.5:7b",
    top_k: int = 5
) -> Dict[str, Any]:
    """End-to-end RAG: retrieve chunks, build prompt, generate answer."""

    connect(database_url)
    similar_chunks = retrieve_similar_chunks(
                                             connection=connect(database_url),
                                             top_k=top_k, 
                                             query_text=question, 
                                             model='all-MiniLM-L6-v2')
    print(similar_chunks)
    prompt = build_prompt(similar_chunks, question)
    response = call_ollama(prompt)

    return response

if __name__ == "__main__":
    question = 'Prompt here'
    print(answer_question(question=question, 
                          database_url=os.getenv('DATABASE_URL')))