import os

from .chunker import chunk_documents
from .embeddings import EmbeddingManager
from .loader import load_pdf
from .vector_store import PineconeVectorStore

def run_ingestion(pdf_path: str, source_name: str):
    print(f"1. Loading PDF from: {pdf_path}...")
    pages = load_pdf(pdf_path)
    print(f"   Loaded {len(pages)} pages.")
    if not pages:
        raise ValueError("The PDF contains no extractable text.")

    print("2. Splitting text into chunks...")
    chunks = chunk_documents(pages, source_name=source_name)
    print(f"   Generated {len(chunks)} chunks.")
    if not chunks:
        raise ValueError("The PDF produced no text chunks.")

    print("3. Generating vector embeddings...")
    embedder = EmbeddingManager()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_texts(texts)

    print("4. Uploading vectors to Pinecone...")
    store = PineconeVectorStore()
    store.upsert_chunks(chunks, embeddings)
    
    print("Ingestion complete successfully!")
    return len(chunks)

if __name__ == "__main__":
    pdf_file = "data/documents/DBMS_Short_Notes_2_Pages.pdf"
    if os.path.exists(pdf_file):
        run_ingestion(pdf_file, source_name=os.path.basename(pdf_file))
    else:
        print(f"Error: File not found at {pdf_file}. Please place your PDF in the data/documents folder.")