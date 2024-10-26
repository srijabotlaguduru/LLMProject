from sentence_transformers import SentenceTransformer

def generate_embeddings(documents):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    # Extract text content from each document
    text_data = [doc.page_content for doc in documents]
    
    # Check if text_data is not empty
    if not text_data:
        raise ValueError("No text data available to generate embeddings.")
    
    embeddings = model.encode(text_data)
    return embeddings
