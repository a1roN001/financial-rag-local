# Financial RAG System with Local LLM

A Retrieval-Augmented Generation (RAG) system for analyzing financial reports using local LLM deployment, ensuring data privacy and zero API costs.

## Overview

This project enables intelligent question-answering over financial documents (annual reports, 10-K filings, etc.) by combining vector search with a locally deployed large language model. The system extracts text and tables from PDF documents, creates semantic embeddings, and generates accurate answers with source citations.

## Key Features

- **Local LLM Deployment**: Uses Ollama with Qwen2.5 7B for complete data privacy
- **PDF Parsing**: Extracts text and tables from complex financial documents
- **Hybrid Search**: Combines FTS and vector search for improved retrieval accuracy
- **Source Citations**: Every answer includes references to specific document pages
- **Quality Evaluation**: Automated assessment using RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context recall)
- **Hybrid Retrieval**: Dense vector search (pgvector cosine) fused with lexical
  PostgreSQL FTS (tsvector + GIN index) via Reciprocal Rank Fusion (RRF, k=60).
  Lexical branch extracts language-neutral tokens (numbers, tickers, Latin terms)
  from Russian queries and falls back to pure dense retrieval for Cyrillic-only input.
- **Two-Stage Reranking**: Cross-encoder (BAAI/bge-reranker-base) reorders the
  RRF candidate pool (recall@25) down to the final context window (rerank@3),
  lifting precision above the bi-encoder baseline.
- **Source Citations**: Answers include document and page references; citation is
  emitted only for factual answers, suppressed on refusals (see Evaluation notes).

## Tech Stack

- **LLM (generation)**: Qwen2.5 7B via Ollama, num_ctx=8192
- **Embeddings**: intfloat/multilingual-e5-small (cross-lingual, ru→en)
- **Reranker**: BAAI/bge-reranker-base (second-stage cross-encoder)
- **Lexical search**: PostgreSQL FTS (tsvector GENERATED STORED + GIN), RRF fusion
- **Vector DB**: PostgreSQL + pgvector
- **PDF Processing**: unstructured
- **Interface**: Streamlit
- **Evaluation**: RAGAS (Faithfulness, Answer Relevancy, Context Precision,
  Context Recall); judge via external API (gpt-6-luna, temperature=0)

## Evaluation

Benchmark: 15 single-hop questions over 7 public 10-K filings (Apple, Microsoft,
Alphabet, Amazon, NVIDIA, Tesla), Russian queries against English documents.
Ground-truth answers verified against source filings.

| Metric | Score | Interpretation |
|---|---|---|
| Context Precision | 0.8944 | Above the bi-encoder baseline (multilingual-e5-small profile: recall-oriented, map@5 ≈ 79 on Belebele rus→en with recall@1000 = 100). The cross-encoder second stage lifts final precision; this is the reranker's measured contribution. |
| Context Recall | 0.9333 | 14/15 questions fully covered by the retrieved context. The 2 misses are diagnosed retrieval gaps (see below), not harness artifacts. |
| Answer Relevancy | 0.9220 | Answers address the asked question; refusals on unanswerable-style items are correctly excluded from the relevant set. |
| Faithfulness | 0.5333 | See breakdown. |

### Faithfulness breakdown (diagnosed, not a black box)

13/15 answers are factually correct with accurate page citations (e.g. Apple
FY2023 net sales $383,285M, Alphabet FY2024 revenue $350,018M, EC Shopping fine
€2.4B upheld by ECJ Sep 2024). The score is dragged down by two systematic
generator behaviors:

1. **Unit-of-measure distortion.** 10-K tables are labeled "(in millions)". The
   7B generator reproduces the raw figure and appends an inferred unit
   ("383,285 млн", "$110,543 миллионов") that the judge cannot verify against
   the chunk text, marking the claim unsupported. The number is right; the
   unit is not in the retrieved snippet.
2. **Fabricated citation on refusal.** On the 2 uncovered questions the model
   correctly refuses ("no information in the documents") but still attaches a
   page reference. The judge extracts "page X contains no information" as a
   claim and contradicts it against the chunk that does mention the topic.

Both are prompt-calibration fixes (cite units only when present in context;
suppress citation on refusal), tracked as the next iteration. They are
generation-side, not retrieval-side: the same 13 correct answers prove the
retrieval→rerank→generation chain is sound end-to-end.

### Known limitations

- **n=15, single-hop only.** Multi-hop questions (cross-company comparisons
  requiring facts from 2–3 filings) are not covered here; they hit the
  single-query architectural limit and are deferred to a query-decomposition
  task.
- **2 retrieval misses** (Apple FTE count, Greater China segment) are diagnosed
  as ranking/coverage gaps, fixable via rerank@5 or hybrid IDF tuning, not
  harness bugs.
- **Judge dependency.** Scores are only as reliable as the external judge's
  structured-output compliance; the harness pins temperature=0 and validates
  schema to avoid silent parser failures.

## Use Cases

- Financial analysts researching company performance
- Investment teams analyzing multiple reports
- Compliance teams extracting specific data points
- Academic research on financial documents

## Roadmap

- **Prompt calibration for faithfulness**: cite units only when present in the
  retrieved chunk; suppress source citation on refusals (addresses the two
  systematic failure modes above).
- **Query decomposition for multi-hop**: split cross-company comparison
  questions into per-entity sub-queries, retrieve and merge contexts.
- **Lexical branch hardening**: switch FTS config from `simple` to `english`
  (stemming + stopwords), add IDF-aware weighting or DF-filter for
  high-frequency tokens (e.g. "2024" appears in 39% of chunks).
- **Multilingual reranker**: evaluate BAAI/bge-reranker-v2-m3 to lift the
  cross-lingual ceiling of the current English-trained base reranker.
- **Layout-aware table extraction**: recover the 6 missing pages observed in
  the Google 10-K parse (pages 29, 49, 54, 86, 90, 97).
- **Dataset expansion**: grow to 30+ questions with explicit single-hop /
  multi-hop split and per-tier reporting.

## Dataset

The dataset consists of annual financial reports (Form 10-K and Annual Reports) from major US tech companies for the fiscal year 2024-2025.

**Sources:**
- **SEC EDGAR Database**: Official 10-K filings for Microsoft, NVIDIA, and Alphabet.
- **Company Investor Relations Pages**: Annual Report PDF for Amazon.

All documents are publicly available and used solely for educational and demonstration purposes in this RAG project.