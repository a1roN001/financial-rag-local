"""
PDF Parser for Financial Reports (10-K).
Extracts text and tables from PDF files using pdfplumber.
Outputs structured JSON and separate CSV files for tables.
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pdfplumber
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_page_data(page) -> Dict[str, Any]:
    """
    Extract text and tables from a single PDF page.
    Handles encoding errors gracefully by replacing problematic characters.
    """
    try:
        text = page.extract_text() or ""
    except Exception as e:
        logger.warning(f"Encoding error on page {page.page_number}: {e}. Using fallback extraction.")
        text = page.extract_text(layout=True) or ""  # Fallback with layout mode

    tables = []
    raw_tables = page.extract_tables()
    
    for table in raw_tables:
        if table and len(table) > 0:  # Skip empty tables
            tables.append({"rows": table})

    return {
        "page_number": page.page_number,
        "text": text.strip(),
        "tables": tables
    }


def save_tables_to_csv(tables: List[Dict], output_path: Path, filename_stem: str):
    """
    Save all extracted tables from a PDF into a single CSV file.
    Each table is separated by an empty row and a header comment.
    """
    if not tables:
        return
        
    csv_path = output_path / f"{filename_stem}_tables.csv"
    
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            for i, table_data in enumerate(tables):
                rows = table_data["rows"]
                if rows:
                    # Add separator between tables
                    if i > 0:
                        writer.writerow([]) 
                    
                    # Write table rows
                    for row in rows:
                        # Handle None values in cells
                        clean_row = [str(cell) if cell is not None else "" for cell in row]
                        writer.writerow(clean_row)
                        
        logger.info(f"💾 Saved {len(tables)} tables to {csv_path.name}")
        
    except Exception as e:
        logger.error(f"Failed to save CSV for {filename_stem}: {e}")


def parse_pdf(pdf_path: Path, output_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Parse a single PDF file. Saves JSON structure and separate CSV for tables.
    Handles corrupted files and encoding issues gracefully.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_filename = f"{pdf_path.stem}.json"
    json_path = output_dir / json_filename

    try:
        logger.info(f"Parsing: {pdf_path.name}")
        pages_data = []
        all_tables = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in tqdm(pdf.pages, desc=f"  Pages ({pdf_path.name})", leave=False):
                try:
                    page_info = extract_page_data(page)
                    pages_data.append(page_info)
                    all_tables.extend(page_info["tables"])
                except Exception as e:
                    logger.error(f"Error processing page {page.page_number} of {pdf_path.name}: {e}")
                    continue  # Skip broken page, continue with next

        result = {
            "filename": pdf_path.name,
            "total_pages": len(pages_data),
            "pages": pages_data
        }

        # Save main JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Save tables separately as CSV
        save_tables_to_csv(all_tables, output_dir, pdf_path.stem)

        logger.info(f"✅ Saved: {json_filename} ({len(pages_data)} pages)")
        return result

    except pdfplumber.utils.PDFSyntaxError:
        logger.error(f"❌ Corrupted PDF file: {pdf_path.name}. Skipping.")
        return None
    except PermissionError:
        logger.error(f"❌ Permission denied reading: {pdf_path.name}. Check file locks.")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error parsing {pdf_path.name}: {type(e).__name__}: {e}")
        return None


def parse_directory(input_dir: str, output_dir: str) -> List[Dict[str, Any]]:
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

    logger.info(f"Done! Successfully parsed {len(results)}/{len(pdf_files)} files")
    return results


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

    print(f"📂 Looking for PDFs in: {RAW_DIR}")
    
    if not RAW_DIR.exists():
        logger.error(f"Directory does not exist: {RAW_DIR}")
        exit(1)
        
    pdf_count = len(list(RAW_DIR.glob("*.pdf")))
    print(f"🔍 Found {pdf_count} PDF file(s)")
    
    if pdf_count == 0:
        logger.warning("No .pdf files found! Check extensions and folder.")
        exit(1)

    parsed_docs = parse_directory(str(RAW_DIR), str(PROCESSED_DIR))

    if parsed_docs:
        print(f"\n📊 Summary:")
        for doc in parsed_docs:
            total_tables = sum(len(p["tables"]) for p in doc["pages"])
            print(f"  • {doc['filename']}: {doc['total_pages']} pages, {total_tables} tables")
    else:
        logger.error("No documents were successfully parsed.")