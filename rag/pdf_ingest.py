"""
pdf_ingest.py
=============
Extracts text from PDFs and chunks them for the RAG pipeline.
Outputs chunks to data/rag/pdf_chunks.jsonl.
"""

import json
import os

# Try to use fitz (PyMuPDF), fallback to pdfplumber
try:
    import fitz
    PDF_LIB = "fitz"
except ImportError:
    try:
        import pdfplumber
        PDF_LIB = "pdfplumber"
    except ImportError:
        print("Error: Neither PyMuPDF (fitz) nor pdfplumber found. Please install one of them.")
        exit(1)

PDF_DIR = "data/raw/pdfs"
OUT_DIR = "data/rag"
OUT_FILE = os.path.join(OUT_DIR, "pdf_chunks.jsonl")

def extract_text(pdf_path):
    text = ""
    if PDF_LIB == "fitz":
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
    elif PDF_LIB == "pdfplumber":
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    return text

def chunk_text(text, max_words=100):
    """Simple chunking by paragraphs, splitting large ones if needed."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    
    current_chunk = []
    current_len = 0
    
    for p in paragraphs:
        words = p.split()
        if len(words) > max_words:
            start = 0
            while start < len(words):
                chunks.append(" ".join(words[start:start+max_words]))
                start += max_words
        else:
            if current_len + len(words) > max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = words
                current_len = len(words)
            else:
                current_chunk.extend(words)
                current_len += len(words)
                
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(PDF_DIR):
        print(f"Error: {PDF_DIR} does not exist.")
        return
        
    all_chunks = []
    for filename in os.listdir(PDF_DIR):
        if not filename.endswith(".pdf"):
            continue
            
        filepath = os.path.join(PDF_DIR, filename)
        print(f"Extracting {filename} using {PDF_LIB}...")
        
        text = extract_text(filepath)
        text_chunks = chunk_text(text, max_words=150)
        
        for i, chunk in enumerate(text_chunks):
            # Format to match what retriever.py expects
            all_chunks.append({
                "chunk_id": f"{filename}_chunk_{i}",
                "department": "General", 
                "text": chunk,
                "question": f"Context from {filename}",
                "answer": chunk,
                "keywords": [],
                "source": "pdf"
            })
            
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    print(f"✓ {len(all_chunks)} PDF chunks written → {OUT_FILE}")

if __name__ == "__main__":
    main()
