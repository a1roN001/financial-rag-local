# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
RERANKER_MODEL_NAME  = "BAAI/bge-reranker-v2-m3"
GENERATOR_MODEL      = "qwen2.5:7b"
JUDGE_MODEL          = "gpt-6-luna"

RECALL_K   = 25    # кандидатов из каждой ветки и в пул реранка
RERANK_K   = 3    # чанков в промпт
FTS_CONFIG = "english"

NUM_CTX       = 8192
GEN_TIMEOUT   = 120
DB_TIMEOUT_MS = 15000
JUDGE_TIMEOUT = 600
EVAL_WORKERS  = 4
RELEVANCY_STRICTNESS = 1

HF_HOME      = f"{os.getenv("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))}"
DATABASE_URL = f"{os.getenv("DATABASE_URL")}"
# SIMILARITY_THRESHOLD намеренно отсутствует: после RRF+rerank отбор по позиции

# Ветки retrieval. Меняя флаги, получаешь тривиальный ablation
# без правки кода: vector-only / vector+bm25 / vector+fts / triple.
USE_VECTOR = True
USE_BM25   = True
RRF_K      = 60      # константа RRF, не магия: вынесена сюда
