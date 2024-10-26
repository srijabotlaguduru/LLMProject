import os
import tempfile

# Set custom temporary directory
tempdir = tempfile.mkdtemp(dir="C:\\Users\\srija\\Temp")
os.environ['TMPDIR'] = tempdir
os.environ['TEMP'] = tempdir
os.environ['TMP'] = tempdir


import streamlit as st
from backend.data_loader import fetch_and_process_data
from backend.embeddings import generate_embeddings
from backend.faiss_index import build_index
from backend.qa_system import ask_question
from sentence_transformers import SentenceTransformer

def main():
    st.title("Web QA System")

    # Sidebar for URL input with multiple input boxes
    st.sidebar.header("Settings")
    st.sidebar.subheader("Enter URLs")
    
    # Session state to keep track of URLs
    if 'url_boxes' not in st.session_state:
        st.session_state.url_boxes = [1]
    if 'urls' not in st.session_state:
        st.session_state.urls = [""]

    # Function to add new URL input box
    def add_url_box():
        st.session_state.url_boxes.append(len(st.session_state.url_boxes) + 1)
        st.session_state.urls.append("")

    # Display URL input boxes
    for i in range(len(st.session_state.url_boxes)):
        st.session_state.urls[i] = st.sidebar.text_input(f'URL {i + 1}', st.session_state.urls[i])

    st.sidebar.button("Add another URL", on_click=add_url_box)

    # Button to fetch data from URLs
    if st.sidebar.button("Fetch Data"):
        if any(st.session_state.urls):
            documents = fetch_and_process_data(st.session_state.urls)
            
            # Debug: Check the content of fetched documents
            st.write(f"Fetched documents: {[doc.page_content[:500] for doc in documents]}")  # Print first 500 characters of each document
            
            try:
                embeddings = generate_embeddings(documents)
                st.session_state.index = build_index(embeddings)
                st.session_state.documents = documents
                st.session_state.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
                st.write(f"Fetched and processed {len(documents)} documents.")
            except ValueError as e:
                st.error(f"Error generating embeddings: {e}")
        else:
            st.sidebar.error("Please enter at least one URL.")

    # Main section for question input and displaying answer
    question = st.text_input("Enter your question:")

    if st.button("Ask"):
        if question:
            if 'index' in st.session_state and 'documents' in st.session_state and 'embedding_model' in st.session_state:
                api_key = st.secrets["google_palm"]["AIzaSyDMj38eAv2Gdq4SrbEAjhV1Afe38yNMQVU"]
                answer = ask_question(question, st.session_state.index, st.session_state.documents, st.session_state.embedding_model, api_key)
                st.write(f"Answer: {answer}")
            else:
                st.error("Please fetch the data first.")
        else:
            st.error("Please enter a question.")

if __name__ == "__main__":
    main()
