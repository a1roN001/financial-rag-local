-- Create the database schema for storing document chunks
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL,
    page_number INT NOT NULL,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    token_count INT NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR(1536) -- Assuming 1536 dimensions for embeddings (e.g., OpenAI embeddings)
);

-- Install the pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create an IVFFlat index for cosine similarity search
CREATE INDEX document_chunks_embedding_idx
ON document_chunks USING ivfflat (embedding)
WITH (lists = 100); -- Adjust 'lists' based on dataset size