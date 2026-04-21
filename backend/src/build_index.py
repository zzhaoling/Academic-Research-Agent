#!/usr/bin/env python
"""Build hybrid retrieval index from PDF files in data/papers."""

import os
import json
from pathlib import Path
from config import Configuration
from services.pdf_parser import parse_pdf
from services.chunker import chunk_text
from services.vector_store import VectorStore
from services.bm25_store import BM25Store

def main():
    config = Configuration.from_env()
    pdf_dir = Path(config.pdf_dir)
    if not pdf_dir.exists():
        print(f"PDF directory {pdf_dir} does not exist. Creating...")
        pdf_dir.mkdir(parents=True)

    all_chunks = []
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found. Please add PDFs to papers/")
        return

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        pages = parse_pdf(str(pdf_path))
        for page in pages:
            chunks = chunk_text(page["text"], config.chunk_size, config.chunk_overlap)
            for chunk_text_str in chunks:
                all_chunks.append({
                    "content": chunk_text_str,
                    "doc": pdf_path.name,
                    "page": page["page"]
                })

    # Save chunk metadata
    chunks_meta_path = Path(config.chunks_metadata_path)
    chunks_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_meta_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_chunks)} chunks to {chunks_meta_path}")

    # Build vector index
    print("Building vector index...")
    vs = VectorStore(config)
    vs.add_chunks(all_chunks)
    print("Vector index built.")

    # Build BM25 index
    print("Building BM25 index...")
    bm = BM25Store()
    bm.build(all_chunks)
    bm.save(config.bm25_index_path)
    print(f"BM25 index saved to {config.bm25_index_path}")

    print("✅ Index building complete.")

if __name__ == "__main__":
    main()