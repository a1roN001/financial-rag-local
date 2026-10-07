"""PostgreSQL persistence for document chunks and their embeddings."""

import re
from typing import Any, Dict, Iterable, List


VECTOR_DIMENSION_PATTERN = re.compile(r"vector\((\d+)\)", re.IGNORECASE)


def connect(database_url: str) -> Any:
    """Open a PostgreSQL connection."""
    import psycopg2

    return psycopg2.connect(database_url)


def vector_literal(values: Iterable[float]) -> str:
    """Format an embedding for PostgreSQL's vector input syntax."""
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def create_schema(connection: Any, table_name: str, dimension: int) -> None:
    """Create pgvector storage and verify an existing table's dimension."""
    from psycopg2 import sql

    table = sql.Identifier(table_name)
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    section VARCHAR(255),
                    page_number INT NOT NULL,
                    chunk_id VARCHAR(255) UNIQUE NOT NULL,
                    token_count INT NOT NULL,
                    text TEXT NOT NULL,
                    is_table BOOLEAN DEFAULT FALSE,
                    embedding VECTOR({dimension}) NOT NULL
                )
                """
            ).format(table=table, dimension=sql.Literal(dimension))
        )
        cursor.execute(
            """
            SELECT format_type(attribute.atttypid, attribute.atttypmod)
            FROM pg_attribute AS attribute
            JOIN pg_class AS relation ON relation.oid = attribute.attrelid
            WHERE relation.relname = %s
              AND attribute.attname = 'embedding'
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
            """,
            (table_name,),
        )
        result = cursor.fetchone()
        if result is None:
            raise RuntimeError(f"Table {table_name!r} has no embedding column")

        match = VECTOR_DIMENSION_PATTERN.search(result[0])
        if match and int(match.group(1)) != dimension:
            raise RuntimeError(
                f"Table {table_name!r} uses VECTOR({match.group(1)}), "
                f"but the model produces {dimension} dimensions. "
                "Use a matching model or recreate the table."
            )

        cursor.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} "
                "ON {table} USING hsnw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_contruction = 64)"
            ).format(
                index=sql.Identifier(f"{table_name}_embedding_idx"),
                table=table,
            )
        )
    connection.commit()


def upsert_chunks(connection: Any, table_name: str, chunks: List[Dict[str, Any]], embeddings: Any) -> None:
    """Insert chunks and update embeddings for chunk IDs already in the table."""
    from psycopg2 import sql
    from psycopg2.extras import execute_values

    rows = [
        (
            str(chunk["doc_id"]),
            int(chunk["page_number"]),
            str(chunk["chunk_id"]),
            int(chunk["token_count"]),
            str(chunk["text"]),
            vector_literal(embedding),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    if not rows:
        return

    query = sql.SQL(
        """
        INSERT INTO {table}
            (document_id, page_number, chunk_id, token_count, text, embedding)
        VALUES %s
        ON CONFLICT (chunk_id) DO UPDATE SET
            document_id = EXCLUDED.document_id,
            page_number = EXCLUDED.page_number,
            token_count = EXCLUDED.token_count,
            text = EXCLUDED.text,
            embedding = EXCLUDED.embedding
        """
    ).format(table=sql.Identifier(table_name))
    with connection.cursor() as cursor:
        execute_values(
            cursor,
            query.as_string(connection),
            rows,
            template="(%s, %s, %s, %s, %s, %s::vector)",
            page_size=len(rows),
        )
    connection.commit()