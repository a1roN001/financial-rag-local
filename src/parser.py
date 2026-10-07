"""
PDF Parser for Financial Reports (10-K).
Extracts text, tables, and structure from PDF files using unstructured.
Outputs structured JSON with metadata for the chunking pipeline.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from unstructured.partition.pdf import partition_pdf

# Настройка Tesseract для Windows (если он установлен по этому пути)
os.environ["PATH"] += os.pathsep + r"C:/Program Files/Tesseract-OCR"
os.environ["OCR_AGENT"] = "tesseract"

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_pdf(pdf_path: Path, output_dir: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Parse a single PDF file using unstructured.
    Extracts elements, identifies sections by titles, and saves to JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Parsing: {pdf_path.name}")
        
        # strategy="fast" использует pdfminer/pdfplumber под капотом, но дает нам категоризацию.
        # Если нужно лучшее качество таблиц, поменяй на strategy="hi_res" (требует больше RAM/GPU)
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="fast",
            infer_table_structure=True
        )
        
        parsed_elements = []
        current_section = "Unknown"
        doc_id = pdf_path.stem
        
        for element in elements:
            # Определяем секцию по заголовкам (Title или Header)
            if element.category in ["Title", "Header"]:
                # Обновляем текущую секцию, если это не колонтитул (проверка по длине или ключевым словам опциональна)
                current_section = str(element).strip()
            
            # Пропускаем служебный мусор
            if element.category in ["Footer", "PageNumber", "EmailAddress", "Image"]:
                continue
            
            parsed_elements.append({
                "doc_id": doc_id,
                "section": current_section,
                "element_type": element.category,
                "text": str(element),  # Таблицы автоматически конвертируются в Markdown
                "page_number": getattr(element.metadata, "page_number", None) or 0,
                "is_table": element.category == "Table"
            })
        
        # Сохранение в JSON
        output_file = output_dir / f"{doc_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed_elements, f, ensure_ascii=False, indent=2)
        
        table_count = sum(1 for el in parsed_elements if el["is_table"])
        logger.info(f"Saved: {output_file.name} ({len(parsed_elements)} elements, {table_count} tables)")
        
        return parsed_elements
        
    except Exception as e:
        logger.error(f"Failed to parse {pdf_path.name}: {type(e).__name__}: {e}")
        return None


def parse_directory(input_dir: str, output_dir: str) -> List[List[Dict[str, Any]]]:
    """Parse all PDF files in the input directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    pdf_files = sorted(input_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    results = []

    for pdf_file in pdf_files:
        result = parse_pdf(pdf_file, output_path)
        if result is not None:
            results.append(result)

    logger.info(f"🏁 Done! Successfully parsed {len(results)}/{len(pdf_files)} files")
    return results


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

    print(f"Looking for PDFs in: {RAW_DIR}")
    
    if not RAW_DIR.exists():
        logger.error(f"Directory does not exist: {RAW_DIR}")
        exit(1)
        
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    if not pdf_files:
        logger.warning("No .pdf files found! Check extensions and folder.")
        exit(1)

    # ЗАПУСК ПАРСИНГА
    parsed_docs = parse_directory(str(RAW_DIR), str(PROCESSED_DIR))

    if parsed_docs:
        print("\n📊 Summary:")
        for doc in parsed_docs:
            total_elements = len(doc)
            total_tables = sum(1 for el in doc if el.get("is_table"))
            print(f"  • {doc[0]['doc_id']}.json: {total_elements} elements, {total_tables} tables")
    else:
        logger.error("No documents were successfully parsed. Check logs for errors.")