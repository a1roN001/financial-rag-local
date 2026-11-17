"""Hybrid retrieval: dense (pgvector) + lexical (BM25), fused via RRF, reranked."""
import logging
from typing import Dict, List, Tuple

from psycopg2 import sql

from config import RECALL_K, RERANK_K, RRF_K, USE_BM25, USE_VECTOR, RERANKER_MODEL_NAME
from bm25_index import get_bm25_index
from db import vector_literal
from sentence_transformers import  CrossEncoder

RERANKER_MODEL = CrossEncoder(RERANKER_MODEL_NAME)

logger = logging.getLogger(__name__)

_VECTOR_SQL = sql.SQL("""
    SELECT chunk_id,
           1 - (embedding <=> %(vec)s::vector) AS similarity,
           ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank
    FROM {table}
    ORDER BY embedding <=> %(vec)s::vector
    LIMIT %(k)s
""")

_META_SQL = sql.SQL("""
    SELECT chunk_id, document_id, page_number, token_count, text, section, is_table
    FROM {table}
    WHERE chunk_id = ANY(%(ids)s)
""")


def _vector_rank(connection, model, query_text, table_name) -> Dict[str, Tuple[float, int]]:
    emb = model.encode(f"query: {query_text}", normalize_embeddings=True)
    out: Dict[str, Tuple[float, int]] = {}
    q = _VECTOR_SQL.format(table=sql.Identifier(table_name))
    with connection.cursor() as cur:
        cur.execute(q, {"vec": vector_literal(emb), "k": RECALL_K})
        for cid, sim, rank in cur.fetchall():
            out[cid] = (float(sim), int(rank))
    return out


def _bm25_rank(query_text) -> Dict[str, int]:
    # Пусто для чисто кириллического вопроса -> RRF деградирует до vector-only
    return {cid: r for cid, r in get_bm25_index().rank(query_text, RECALL_K)}


def retrieve_similar_chunks(connection, model, query_text, top_k, table_name="document_chunks"):
    if not query_text.strip():
        raise ValueError("query_text must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")

    vec = _vector_rank(connection, model, query_text, table_name) if USE_VECTOR else {}
    lex = _bm25_rank(query_text) if USE_BM25 else {}

    union = set(vec) | set(lex)
    if not union:
        return []

    fused: List[Tuple[str, float]] = []
    for cid in union:
        score = 0.0
        if cid in vec:
            score += 1.0 / (RRF_K + vec[cid][1])
        if cid in lex:
            score += 1.0 / (RRF_K + lex[cid])
        fused.append((cid, score))
    fused.sort(key=lambda x: x[1], reverse=True)
    fused = fused[:RECALL_K]

    ids = [cid for cid, _ in fused]
    mq = _META_SQL.format(table=sql.Identifier(table_name))
    with connection.cursor() as cur:
        cur.execute(mq, {"ids": ids})
        cols = [d[0] for d in cur.description]
        meta = {row[0]: dict(zip(cols, row)) for row in cur.fetchall()}

    results: List[Dict] = []
    for cid, hscore in fused:
        row = meta.get(cid)
        if row is None:
            continue
        row["similarity"] = vec[cid][0] if cid in vec else None  # None для BM25-only
        row["hybrid_score"] = hscore
        results.append(row)

    for r in results:
        sim = r["similarity"]
        sim_str = f"{sim:.4f}" if sim is not None else " None"
        logger.info(
            "Retrieved %s from %s p.%s (sim=%s, hybrid=%.4f): %s",
            r["chunk_id"], r["document_id"], r["page_number"],
            sim_str, r["hybrid_score"], r["text"][:160].replace("\n", " "),
        )
    return results

def rerank_chunks(query, chunks, top_k):
    if not chunks:
        return []
    scores = RERANKER_MODEL.predict([[query, c["text"]] for c in chunks])
    scored = [dict(c, rerank_score=float(s)) for c, s in zip(chunks, scores)]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]