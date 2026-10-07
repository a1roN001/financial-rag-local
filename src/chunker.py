"""
Text Chunker for RAG Pipeline.
Processes flat lists of elements from unstructured parser.
Creates semantic chunks, keeping tables atomic and preserving section metadata.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import tiktoken
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_chunk_metadata(elements: List[Dict], chunk_index: int, doc_id: str, tokenizer: Any) -> Dict[str, Any]:
    """
    Aggregates text and metadata from a list of elements into a single chunk.
    """
    if not elements:
        return {}

    # Берем метаданные из первого элемента в группе для консистентности
    first_el = elements[0]
    
    # Собираем текст, добавляя двойной перенос строки между разными элементами для сохранения структуры
    combined_text = "\n\n".join(el["text"] for el in elements if el["text"].strip())
    
    # Если после очистки текст пустой, возвращаем None
    if not combined_text.strip():
        return None

    token_count = len(tokenizer.encode(combined_text))
    
    # Формируем chunk_id. Если это таблица, добавляем маркер 'tbl'
    chunk_type = "tbl" if first_el.get("is_table") else "chk"
    chunk_id = f"{doc_id}_{chunk_type}_{chunk_index}"

    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "section": first_el.get("section", "Unknown"),
        "page_number": first_el.get("page_number", 0),
        "text": combined_text,
        "token_count": token_count,
        "is_table": first_el.get("is_table", False)
    }


def chunk_elements(
    elements: List[Dict[str, Any]], 
    chunk_size: int = 500, 
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Chunk a flat list of parsed elements.
    Keeps tables atomic. Accumulates text elements up to chunk_size.
    """
    if not elements:
        return []

    tokenizer = tiktoken.get_encoding("cl100k_base")
    doc_id = elements[0].get("doc_id", "unknown_doc")
    
    chunks = []
    current_elements = []
    current_tokens = 0
    chunk_index = 0

    for element in elements:
        text = element.get("text", "")
        if not text.strip():
            continue

        # ПРАВИЛО 1: Таблицы всегда атомарны. 
        # Если встречаем таблицу, сначала сбрасываем накопленный текст, потом добавляем таблицу как отдельный чанк.
        if element.get("is_table"):
            if current_elements:
                chunk_data = build_chunk_metadata(current_elements, chunk_index, doc_id, tokenizer)
                if chunk_data:
                    chunks.append(chunk_data)
                    chunk_index += 1
                current_elements = []
                current_tokens = 0

            # Добавляем таблицу как отдельный чанк
            table_chunk = build_chunk_metadata([element], chunk_index, doc_id, tokenizer)
            if table_chunk:
                chunks.append(table_chunk)
                chunk_index += 1
            continue

        # ПРАВИЛО 2: Накопление текста
        element_tokens = len(tokenizer.encode(text))

        # Если добавление элемента превышает лимит, и у нас уже что-то есть в буфере
        if current_tokens + element_tokens > chunk_size and current_elements:
            # Сохраняем текущий чанк
            chunk_data = build_chunk_metadata(current_elements, chunk_index, doc_id, tokenizer)
            if chunk_data:
                chunks.append(chunk_data)
                chunk_index += 1
            
            # ПРАВИЛО 3: Overlap (перекрытие)
            # Вместо разрезания токенов (что ломает Markdown), мы берем последний элемент 
            # предыдущего чанка и начинаем новый с него, если он влезает в overlap.
            # Для простоты и надежности: начинаем новый чанк с текущего элемента, 
            # но в реальном продакшене здесь можно добавить логику сохранения 1-2 последних элементов.
            current_elements = [element]
            current_tokens = element_tokens
        else:
            current_elements.append(element)
            current_tokens += element_tokens

    # Сброс остатка
    if current_elements:
        chunk_data = build_chunk_metadata(current_elements, chunk_index, doc_id, tokenizer)
        if chunk_data:
            chunks.append(chunk_data)

    return chunks


def process_directory(processed_dir: str, output_dir: str, chunk_size: int = 500, overlap: int = 50):
    """
    Process all JSON files from the parser output and create chunks.
    """
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(processed_path.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {processed_dir}. Run parser first.")
        return []
        
    logger.info(f"Found {len(json_files)} processed documents to chunk")
    total_chunks = 0
    
    for json_file in tqdm(json_files, desc="Chunking documents"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                elements = json.load(f)
                
            if not isinstance(elements, list):
                logger.warning(f"Skipping {json_file.name}: expected a flat list of elements")
                continue
                
            doc_id = json_file.stem
            file_chunks = chunk_elements(elements, chunk_size=chunk_size, overlap=overlap)
            
            # Сохраняем чанки для этого документа
            chunk_output = output_path / f"{doc_id}_chunks.json"
            with open(chunk_output, "w", encoding="utf-8") as f:
                json.dump(file_chunks, f, ensure_ascii=False, indent=2)
                
            total_chunks += len(file_chunks)
            logger.info(f"Created {len(file_chunks)} chunks from {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to chunk {json_file.name}: {e}")
            continue
            
    logger.info(f"🏁 Total chunks created across all documents: {total_chunks}")
    return total_chunks


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
    
    print(f"📂 Processing documents from: {PROCESSED_DIR}")
    
    if not PROCESSED_DIR.exists():
        logger.error(f"Processed directory does not exist: {PROCESSED_DIR}")
        exit(1)
        
    process_directory(
        str(PROCESSED_DIR), 
        str(CHUNKS_DIR),
        chunk_size=500,
        overlap=50
    )