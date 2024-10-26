from langchain.document_loaders import UnstructuredURLLoader

def fetch_and_process_data(urls):
    loader = UnstructuredURLLoader(urls)
    documents = loader.load()
    
    # Debug: Print documents to check their content
    for doc in documents:
        print(f"Document content: {doc.page_content[:500]}")  # Print first 500 characters

    return documents
