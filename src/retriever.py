"""Similarity search over stored document chunk embeddings."""

import os
os.environ["HF_HOME"] = "C:\\hf_cache"
import re
import logging
from typing import Any, Dict, List
from psycopg2 import sql
from db import vector_literal
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
RERANKER_MODEL = CrossEncoder("BAAI/bge-reranker-v2-m3")
DB_TIMEOUT_MS = '5000'
load_dotenv()
COLUMNS = sql.SQL(
    "t.document_id, t.page_number, t.chunk_id, t.token_count, "
    "t.text, t.section, t.is_table"
)
def extract_fts_tokens(user_query: str) -> str:
    """Извлекает числа, годы, тикеры и латинские слова для FTS."""
    tokens = re.findall(r'\b[A-Za-z0-9]+(?:[.,][0-9]+)?\b', user_query)
    # Убираем дубликаты с сохранением порядка
    unique_tokens = list(dict.fromkeys(tokens))
    return " OR ".join(unique_tokens)

def retrieve_similar_chunks(
    connection: Any,
    model,
    query_text: str,
    top_k: int,
    table_name: str = "document_chunks"
) -> List[Dict[str, Any]]:
    """Retrieve the top-k chunks using Hybrid Search (Vector + FTS) or fallback to Vector."""
    if not query_text.strip():
        raise ValueError("query_text must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")

    query_embedding = model.encode(f"query: {query_text}", normalize_embeddings=True)
    query_vector = vector_literal(query_embedding)

    # Извлекаем language-neutral токены из исходного вопроса
    fts_query = extract_fts_tokens(query_text)

    if fts_query:
        # ВЕТКА 1: Есть цифры/латиница -> запускаем гибридный поиск (Vector + FTS)
        query = sql.SQL("""
    WITH semantic_search AS (
        SELECT
            chunk_id,
            1 - (embedding <=> %(vec)s::vector) AS similarity,
            ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank
        FROM {table}
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(k)s
    ),
    keyword_search AS (
        SELECT
            chunk_id,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(fts, websearch_to_tsquery(%(cfg)s, %(fts)s)) DESC
            ) AS rank
        FROM {table}
        WHERE fts @@ websearch_to_tsquery(%(cfg)s, %(fts)s)
        ORDER BY ts_rank_cd(fts, websearch_to_tsquery(%(cfg)s, %(fts)s)) DESC
        LIMIT %(k)s
    ),
    candidates AS (
        SELECT chunk_id FROM semantic_search
        UNION
        SELECT chunk_id FROM keyword_search
    )
    SELECT
        {columns},
        s.similarity,
        (COALESCE(1.0 / (60 + s.rank), 0.0)
         + COALESCE(1.0 / (60 + k.rank), 0.0)) AS hybrid_score
    FROM candidates c
    JOIN {table} t          ON t.chunk_id = c.chunk_id
    LEFT JOIN semantic_search s ON s.chunk_id = c.chunk_id
    LEFT JOIN keyword_search  k ON k.chunk_id = c.chunk_id
    ORDER BY hybrid_score DESC
    LIMIT %(k)s
""").format(table=sql.Identifier(table_name), columns=COLUMNS)

        params = {"vec": query_vector, "fts": fts_query, "cfg": 'english', "k": 25}
    else:
        # ВЕТКА 2: Чисто русский вопрос без цифр и латиницы -> обычный векторный поиск
        query = sql.SQL("""
    SELECT {columns},
           1 - (t.embedding <=> %(vec)s::vector) AS similarity,
           1 - (t.embedding <=> %(vec)s::vector) AS hybrid_score
    FROM {table} t
    ORDER BY t.embedding <=> %(vec)s::vector
    LIMIT %(k)s
""").format(table=sql.Identifier(table_name), columns=COLUMNS)

        params = {"vec": query_vector, "k": 25}

    from psycopg2.extras import RealDictCursor

    with connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(f"SET statement_timeout = {DB_TIMEOUT_MS};")
        cursor.execute(query, params)
        results = [dict(r) for r in cursor.fetchall()]

    for r in results:
        sim = r["similarity"]
        sim_str = f"{sim:.4f}" if sim is not None else " None"
        logger.info(
            "Retrieved %s from %s p.%s (sim=%s, hybrid=%.4f, fts=%r): %s",
            r["chunk_id"], r["document_id"], r["page_number"],
            sim_str, float(r["hybrid_score"]), fts_query,
            r["text"][:160].replace("\n", " "),
    )
    return results


def rerank_chunks(query, chunks, top_k):
    if not chunks:
        return []
    scores = RERANKER_MODEL.predict([[query, c["text"]] for c in chunks])
    scored = [dict(c, rerank_score=float(s)) for c, s in zip(chunks, scores)]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]
