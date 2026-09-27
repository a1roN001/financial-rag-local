"""
Text Chunker for RAG Pipeline.
Splits processed text into semantic chunks with overlap and sentence-aware boundaries.
Outputs structured JSON with metadata for vector database ingestion.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import tiktoken
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using basic punctuation rules.
    Preserves sentence integrity to avoid splitting in the middle.
    """
    if not text.strip():
        return []
    
    # Simple but effective sentence splitter for financial docs
    # Handles common abbreviations and decimal numbers
    sentences = []
    current_sentence = ""
    
    words = text.split()
    for word in words:
        current_sentence += word + " "
        
        # Check if word ends a sentence
        if word.endswith(('.', '!', '?')) and not any(
            word.lower().endswith(abbr) for abbr in 
            ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'inc.', 'ltd.', 'co.', 'vs.', 'etc.', 'u.s.', 'e.g.', 'i.e.']
        ):
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    # Add remaining text as final sentence
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
        
    return sentences


def create_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    doc_id: str = "",
    page_number: int = 0
) -> List[Dict[str, Any]]:
    """
    Create semantic chunks from text with specified size and overlap.
    Ensures sentences are never split in the middle.
    
    Args:
        text: Input text to chunk
        chunk_size: Target number of tokens per chunk
        overlap: Number of overlapping tokens between chunks
        doc_id: Document identifier for metadata
        page_number: Page number within document
        
    Returns:
        List of chunk dictionaries with metadata
    """
    if not text.strip():
        return []
        
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Standard for GPT-4/Llama3
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
        
    chunks = []
    current_chunk_tokens = []
    current_chunk_text = ""
    chunk_id = 0
    
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
        
        # If adding this sentence exceeds chunk size, save current chunk
        if len(current_chunk_tokens) + len(sentence_tokens) > chunk_size and current_chunk_tokens:
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_p{page_number}_c{chunk_id}",
                "page_number": page_number,
                "text": current_chunk_text.strip(),
                "token_count": len(current_chunk_tokens)
            })
            chunk_id += 1
            
            # Start new chunk with overlap
            # Calculate how many tokens to keep for overlap
            overlap_tokens = min(overlap, len(current_chunk_tokens))
            if overlap_tokens > 0:
                # Keep last N tokens from previous chunk
                overlap_text = tokenizer.decode(current_chunk_tokens[-overlap_tokens:])
                current_chunk_tokens = current_chunk_tokens[-overlap_tokens:]
                current_chunk_text = overlap_text + " " + sentence
            else:
                current_chunk_tokens = sentence_tokens
                current_chunk_text = sentence
        else:
            current_chunk_tokens.extend(sentence_tokens)
            current_chunk_text += sentence + " "
    
    # Save final chunk
    if current_chunk_tokens:
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_p{page_number}_c{chunk_id}",
            "page_number": page_number,
            "text": current_chunk_text.strip(),
            "token_count": len(current_chunk_tokens)
        })
        
    return chunks


def chunk_processed_documents(processed_dir: str, output_dir: str, chunk_size: int = 500, overlap: int = 50):
    """
    Process all JSON files from parser output and create chunks.
    
    Args:
        processed_dir: Path to directory with parsed JSON files
        output_dir: Path to save chunked JSON files
        chunk_size: Target tokens per chunk
        overlap: Overlap tokens between chunks
    """
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(processed_path.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {processed_dir}. Run parser first.")
        return []
        
    logger.info(f"Found {len(json_files)} processed documents to chunk")
    all_chunks = []
    
    for json_file in tqdm(json_files, desc="Chunking documents"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
                
            doc_id = json_file.stem
            file_chunks = []
            
            for page in doc_data.get("pages", []):
                page_text = page.get("text", "")
                page_num = page.get("page_number", 0)
                
                page_chunks = create_chunks(
                    text=page_text,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    doc_id=doc_id,
                    page_number=page_num
                )
                file_chunks.extend(page_chunks)
            
            # Save chunks for this document
            chunk_output = output_path / f"{doc_id}_chunks.json"
            with open(chunk_output, "w", encoding="utf-8") as f:
                json.dump(file_chunks, f, ensure_ascii=False, indent=2)
                
            all_chunks.extend(file_chunks)
            logger.info(f"Created {len(file_chunks)} chunks from {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to chunk {json_file.name}: {e}")
            continue
            
    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
    
    print(f"Processing documents from: {PROCESSED_DIR}")
    
    if not PROCESSED_DIR.exists():
        logger.error(f"Processed directory does not exist: {PROCESSED_DIR}")
        exit(1)
        
    chunks = chunk_processed_documents(
        str(PROCESSED_DIR), 
        str(CHUNKS_DIR),
        chunk_size=500,
        overlap=50
    )
    
    if chunks:
        print(f"\nSummary:")
        print(f"  Total chunks: {len(chunks)}")
        avg_tokens = sum(c["token_count"] for c in chunks) / len(chunks)
        print(f"  Average chunk size: {avg_tokens:.0f} tokens")