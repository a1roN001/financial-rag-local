"""Create chunk embeddings and load them into PostgreSQL with pgvector."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

from db import connect, create_schema, upsert_chunks
from retriever import retrieve_similar_chunks

load_dotenv()

logger = logging.getLogger(__name__)

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
        with connect(args.database_url) as connection:
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

    logger.info("Connecting to PostgreSQL")
    with connect(args.database_url) as connection:
        create_schema(connection, args.table, dimension)
        upsert_chunks(connection, args.table, chunks, embeddings)

    logger.info("Loaded %d chunks with %d-dimensional embeddings", len(chunks), dimension)


if __name__ == "__main__":
    main()