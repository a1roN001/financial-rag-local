"""Create chunk embeddings and load them into PostgreSQL with pgvector."""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)
VECTOR_DIMENSION_PATTERN = re.compile(r"vector\((\d+)\)", re.IGNORECASE)


def load_chunks(chunks_dir: Path) -> List[Dict[str, Any]]:
    """Load and validate chunk records from all JSON files in a directory."""
    chunks: List[Dict[str, Any]] = []
    for chunk_file in sorted(chunks_dir.glob("*_chunks.json")):
        with chunk_file.open("r", encoding="utf-8") as file:
            records = json.load(file)

        if not isinstance(records, list):
            logger.warning("Skipping %s: expected a JSON list", chunk_file.name)
            continue

        for record in records:
            if not isinstance(record, dict):
                continue
            required_fields = ("doc_id", "chunk_id", "page_number", "token_count", "text")
            if not all(field in record for field in required_fields):
                logger.warning("Skipping incomplete chunk in %s", chunk_file.name)
                continue
            if str(record["text"]).strip():
                chunks.append(record)

    unique_chunks = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    return list(unique_chunks.values())


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
                    page_number INT NOT NULL,
                    chunk_id VARCHAR(255) UNIQUE NOT NULL,
                    token_count INT NOT NULL,
                    text TEXT NOT NULL,
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
                "ON {table} USING ivfflat (embedding vector_cosine_ops) "
                "WITH (lists = 100)"
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


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=project_root / "data" / "chunks",
        help="Directory containing *_chunks.json files",
    )
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--table", default="document_chunks")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--database-url", default=os.getenv("postgresql://postgres:1234@localhost:5432/Vectors"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL is required, for example postgresql://user:password@localhost/db")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be greater than zero")
    if not args.chunks_dir.is_dir():
        raise SystemExit(f"Chunks directory does not exist: {args.chunks_dir}")

    from sentence_transformers import SentenceTransformer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    chunks = load_chunks(args.chunks_dir)
    if not chunks:
        logger.info("No chunks found in %s", args.chunks_dir)
        return

    logger.info("Loading embedding model: %s", args.model)
    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    dimension = len(embeddings[0])

    import psycopg2

    logger.info("Connecting to PostgreSQL")
    with psycopg2.connect(args.database_url) as connection:
        create_schema(connection, args.table, dimension)
        upsert_chunks(connection, args.table, chunks, embeddings)

    logger.info("Loaded %d chunks with %d-dimensional embeddings", len(chunks), dimension)


if __name__ == "__main__":
    main()