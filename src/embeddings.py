"""Create chunk embeddings and load them into PostgreSQL with pgvector."""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List
from dotenv import load_dotenv

load_dotenv()

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
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--query", help="Retrieve similar chunks for a question instead of indexing")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL is required, for example postgresql://user:password@localhost/db")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be greater than zero")
    if args.top_k < 1:
        raise SystemExit("--top-k must be greater than zero")
    if not args.query and not args.chunks_dir.is_dir():
        raise SystemExit(f"Chunks directory does not exist: {args.chunks_dir}")

    from sentence_transformers import SentenceTransformer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.query:
        logger.info("Loading embedding model: %s", args.model)
        model = SentenceTransformer(args.model)
        import psycopg2

        with psycopg2.connect(args.database_url) as connection:
            results = retrieve_similar_chunks(
                connection,
                model,
                args.query,
                table_name=args.table,
                top_k=args.top_k,
            )
        for result in results:
            print(
                f"[{result['similarity']:.4f}] {result['document_id']} "
                f"page {result['page_number']} ({result['chunk_id']})\n"
                f"{result['text']}\n"
            )
        return

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