"""
RAGAS Evaluation Pipeline for Financial RAG System.
Evaluates end-to-end RAG quality: retrieval + generation.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
answer_relevancy.strictness = 1
from tqdm import tqdm
from langchain_openai import ChatOpenAI

from db import connect
from retriever import retrieve_similar_chunks
from rag import build_prompt, call_ollama
from retriever import rerank_chunks
from config import EMBEDDING_MODEL_NAME, GENERATOR_MODEL, JUDGE_MODEL, RERANKER_MODEL_NAME
from ragas.run_config import RunConfig
from wrapper import CustomE5Embeddings
from sentence_transformers import SentenceTransformer, CrossEncoder 
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBEDDING_MODEL_EVAL = CustomE5Embeddings(EMBEDDING_MODEL, model_name=EMBEDDING_MODEL_NAME)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATASET_PATH = PROJECT_ROOT / "data" / "single_hop_questions.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation_results.json"
load_dotenv()

JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL")
def load_test_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Test dataset not found at {path}. "
            "Create it first with 20-30 questions."
        )
    
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    logger.info(f"Loaded {len(dataset)} test questions from {path.name}")
    return dataset


def retrieve_context(question: str, top_k: int) -> list[dict]:
    with connect(f"{os.getenv("DATABASE_URL")}") as conn:
        pool = retrieve_similar_chunks(conn, EMBEDDING_MODEL , question, top_k)
    return rerank_chunks(question, pool, top_k=3)


def generate_rag_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return CANONICAL_REFUSAL

    prompt = build_prompt(contexts, question)
    raw_answer = call_ollama(prompt)
    return sanitize_answer(raw_answer)


def run_evaluation(
    test_dataset: List[Dict[str, Any]],
    top_k: int,
    model_name: str = GENERATOR_MODEL
) -> Dict[str, Any]:
    questions = []
    answers = []
    contexts_for_ragas = []  # <-- Переименовали для ясности (Ragas нужен List[str])
    references = []  
    similarities = []
    
    logger.info(f"Starting evaluation on {len(test_dataset)} questions...")
    
    for item in tqdm(test_dataset, desc="Evaluating"):
        question = item["question"]
        ground_truth = item.get("ground_truth_answer") or item.get("references", "No ground truth")
        
        try:
            # 1. Получаем списком СЛОВАРЕЙ (для build_prompt)
            retrieved_chunks = retrieve_context(question, top_k=top_k)
            
            # 2. Генерируем ответ (передаем словари)
            answer = generate_rag_answer(question, retrieved_chunks)
            
            # 3. Для Ragas извлекаем ТОЛЬКО тексты из этих словарей
            ragas_contexts = [chunk.get("text", "") for chunk in retrieved_chunks]
            
            questions.append(question)
            answers.append(answer)
            contexts_for_ragas.append(ragas_contexts) # <-- Ragas получает List[str]
            references.append(ground_truth) 
            
            # Логируем среднюю схожесть для красоты отчета
            if retrieved_chunks:
                avg_rr = sum(c.get('rerank_score', 0.0) for c in retrieved_chunks) / len(retrieved_chunks)
                similarities.append(f"rr={avg_rr:.2f}")
            else:
                similarities.append("NO_CONTEXT")
            
        except Exception as e:
            logger.error("Failed on %r: %s", question[:50], e, exc_info=True)  # ← traceback
            questions.append(question)
            answers.append("ERROR: Failed to generate answer")
            contexts_for_ragas.append(ragas_contexts if 'ragas_contexts' in locals() else [])  # ← сохранить
            references.append(ground_truth)
            similarities.append("ERROR")
    
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_for_ragas,  # <-- Передали правильный формат для Ragas
        "reference": references
    })
    
        # === НАСТРОЙКА МОДЕЛЕЙ ДЛЯ ОЦЕНКИ ===
    
    # 1. Проверка ключа Groq
    from langchain_ollama import ChatOllama
    
    # Локальная модель успешно переваривает внутренние запросы Ragas, 
    # которые Groq строго блокирует.
    external_llm = ChatOpenAI(
    model=JUDGE_MODEL,          # V4.1-Flash, не deepseek-chat!
    base_url=JUDGE_BASE_URL,
    api_key=os.environ["JUDJE_API_KEY"],
    temperature=0.0,
)


    ragas_run_config = RunConfig(
        max_workers=1,          # Строго 1 запрос за раз!
        timeout=600,            # Ждем до 10 минут на каждую метрику
        max_retries=2           # Повтор при случайном сбое парсинга JSON
    )

    logger.info("Running RAGAS metrics calculation...")
    results = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=external_llm,
        embeddings=EMBEDDING_MODEL_EVAL,
        run_config=ragas_run_config,
    )
    
    scores_df = pd.DataFrame(results.scores)
    metrics_dict = scores_df.mean().to_dict()
    metrics_dict = {metric: float(score) for metric, score in metrics_dict.items()}

    report = {
        "metadata": {
            "total_questions": len(test_dataset),
            "top_k": top_k,
            "similarity_threshold": 0.4,
            "llm_model": model_name,
            "embedding_model": EMBEDDING_MODEL.get_config_dict().get("name", "unknown") if hasattr(EMBEDDING_MODEL, 'get_config_dict') else "intfloat/multilingual-e5-small"
        },
        "metrics": metrics_dict,
        "detailed_results": [
            {
                "question": q,
                "answer": a,
                "reference": gt,
                "num_contexts": len(ctx),
                "avg_similarity": sim
            }
            for q, a, gt, ctx, sim in zip(
                questions, answers, references, contexts_for_ragas, similarities
            )
        ]
    }
    
    return report


def save_results(report: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {path}")


def print_summary(report: Dict[str, Any]) -> None:
    metrics = report["metrics"]
    print("\n" + "="*60)
    print("RAGAS EVALUATION SUMMARY")
    print("="*60)
    for metric, score in metrics.items():
        print(f"  {metric:25s}: {score:.4f}")
    print("="*60)
    
    thresholds = {
        "faithfulness": 0.75,
        "answer_relevancy": 0.80,
        "context_precision": 0.70,
        "context_recall": 0.70
    }
    
    print("\nWarnings:")
    warnings_found = False
    for metric, threshold in thresholds.items():
        if metric in metrics and metrics[metric] < threshold:
            print(f" {metric} ({metrics[metric]:.4f}) below threshold ({threshold})")
            warnings_found = True
            
    if not warnings_found:
        print(" All metrics are above thresholds!")
    print()


def main():
    # Загружаем тестовый набор
    test_dataset = load_test_dataset(TEST_DATASET_PATH)
    
    # Запускаем оценку
    report = run_evaluation(test_dataset, top_k=25)
    
    # Сохраняем и выводим
    save_results(report, RESULTS_PATH)
    print_summary(report)


if __name__ == "__main__":
    main()
