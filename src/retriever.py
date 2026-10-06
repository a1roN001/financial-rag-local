"""Similarity search over stored document chunk embeddings."""

import logging
from typing import Any, Dict, List

from db import vector_literal


logger = logging.getLogger(__name__)


def retrieve_similar_chunks(
    connection: Any,
    model: Any,
    query_text: str,
    table_name: str = "document_chunks",
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve the top-k chunks ranked by cosine similarity to a query."""
    if not query_text.strip():
        raise ValueError("query_text must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")

    query_embedding = model.encode(query_text, normalize_embeddings=True)
    query_vector = vector_literal(query_embedding)

    from psycopg2 import sql

    query = sql.SQL(
        """
        SELECT document_id, page_number, chunk_id, token_count, text,
               1 - (embedding <=> %s::vector) AS similarity
        FROM {table}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
    ).format(table=sql.Identifier(table_name))
    with connection.cursor() as cursor:
        cursor.execute(query, (query_vector, query_vector, top_k))
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

    for result in results:
        logger.info(
            "Retrieved chunk %s from %s page %s (similarity=%.4f): %s",
            result["chunk_id"],
            result["document_id"],
            result["page_number"],
            result["similarity"],
            result["text"][:160].replace("\n", " "),
        )
    return results