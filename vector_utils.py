import numpy as np
import faiss
import json
import os
import hashlib
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, VECTOR_STORE_DIR

def get_vector_store_path(url):
    """Generate unique file path for vector store based on URL"""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return VECTOR_STORE_DIR / f"{url_hash}.index", VECTOR_STORE_DIR / f"{url_hash}.json"

def vector_store_exists(url):
    """Check if vector store exists for given URL"""
    index_path, chunks_path = get_vector_store_path(url)
    return index_path.exists() and chunks_path.exists()

def create_vector_store(text, embedding_model, url):
    """Create and save vector store to local files"""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_text(text)
    
    # Generate embeddings
    embeddings = embedding_model.encode(chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    
    # Save to disk
    index_path, chunks_path = get_vector_store_path(url)
    faiss.write_index(index, str(index_path))
    with open(chunks_path, 'w') as f:
        json.dump(chunks, f)
    
    return index, chunks

def load_vector_store(url):
    """Load vector store from local files"""
    index_path, chunks_path = get_vector_store_path(url)
    index = faiss.read_index(str(index_path))
    with open(chunks_path, 'r') as f:
        chunks = json.load(f)
    return index, chunks

def retrieve_context(query, vector_store, chunks, embedding_model):
    """Retrieve relevant context from vector store"""
    query_embedding = embedding_model.encode([query])
    distances, indices = vector_store.search(query_embedding.astype(np.float32), TOP_K)
    context_chunks = [chunks[i] for i in indices[0]]
    return "\n\n".join(context_chunks)






