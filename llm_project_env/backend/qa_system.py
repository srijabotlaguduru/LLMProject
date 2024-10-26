import numpy as np
import requests

def ask_question(question, index, documents, embedding_model, api_key):
    # Find the most similar document using FAISS
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype('float32')
    _, I = index.search(question_embedding, 1)
    most_similar_doc_index = I[0][0]
    
    # Get the most similar document's text
    most_similar_doc = documents[most_similar_doc_index].page_content
    
    # Use Google PALM API to get the answer
    url = f"https://palm-api-url/answer"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "question": question,
        "context": most_similar_doc
    }
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    return result.get("answer", "No answer found.")
