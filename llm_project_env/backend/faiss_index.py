# backend/faiss_index.py
import faiss
import numpy as np

def build_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    embedding_matrix = np.array(embeddings).astype('float32')
    index.add(embedding_matrix)
    return index
