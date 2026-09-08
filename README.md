# Financial RAG System with Local LLM

A Retrieval-Augmented Generation (RAG) system for analyzing financial reports using local LLM deployment, ensuring data privacy and zero API costs.

## Overview

This project enables intelligent question-answering over financial documents (annual reports, 10-K filings, etc.) by combining vector search with a locally deployed large language model. The system extracts text and tables from PDF documents, creates semantic embeddings, and generates accurate answers with source citations.

## Key Features

- **Local LLM Deployment**: Uses Ollama with Llama 3.1 8B for complete data privacy
- **PDF Parsing**: Extracts text and tables from complex financial documents
- **Hybrid Search**: Combines BM25 and vector search for improved retrieval accuracy
- **Source Citations**: Every answer includes references to specific document pages
- **Quality Evaluation**: Automated assessment using RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision)

## Tech Stack

- **LLM**: Llama 3.1 8B via Ollama
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: PostgreSQL
- **PDF Processing**: PyMuPDF, pdfplumber
- **Framework**: LangChain
- **Interface**: Streamlit
- **Evaluation**: RAGAS

## Results

- Faithfulness Score: 0.87
- Answer Relevancy: 0.91
- Context Precision: 0.83

## Use Cases

- Financial analysts researching company performance
- Investment teams analyzing multiple reports
- Compliance teams extracting specific data points
- Academic research on financial documents

## Future Improvements

- Enhanced table extraction with layout-aware parsing
- Multi-document comparison and summarization
- Support for additional document formats (DOCX, XLSX)
- Deployment optimization with model quantization

## Dataset

The dataset consists of annual financial reports (Form 10-K and Annual Reports) from major US tech companies for the fiscal year 2024-2025.

**Sources:**
- **SEC EDGAR Database**: Official 10-K filings for Microsoft, NVIDIA, and Alphabet.
- **Company Investor Relations Pages**: Annual Report PDF for Amazon.

All documents are publicly available and used solely for educational and demonstration purposes in this RAG project.