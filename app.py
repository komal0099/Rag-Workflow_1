# app.py

# Install Required Libraries:
# pip install streamlit pypdf sentence-transformers faiss-cpu numpy

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------
# Title
# -------------------------------

st.title("RAG PDF Chatbot")

st.write("Upload a PDF and ask questions from it.")

# -------------------------------
# Upload PDF
# -------------------------------

uploaded_file = st.file_uploader("Upload PDF File", type="pdf")

if uploaded_file is not None:

    # -------------------------------
    # Read PDF
    # -------------------------------

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    st.success("PDF Loaded Successfully")

    # -------------------------------
    # Create Chunks
    # -------------------------------

    chunks = []

    chunk_size = 500

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    st.write("Total Chunks:", len(chunks))

    # -------------------------------
    # Load Embedding Model
    # -------------------------------

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # -------------------------------
    # Create Embeddings
    # -------------------------------

    embeddings = model.encode(chunks)

    # -------------------------------
    # Store in FAISS Database
    # -------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    st.success("FAISS Vector Database Created")

    # -------------------------------
    # Ask Question
    # -------------------------------

    query = st.text_input("Ask a Question")

    if query:

        # -------------------------------
        # Search Relevant Chunks
        # -------------------------------

        query_embedding = model.encode([query])

        D, I = index.search(np.array(query_embedding), k=3)

        st.subheader("Relevant Answers")

        for idx in I[0]:
            st.write(chunks[idx])
            st.write("--------------------------------------------------") 
requirement.txt
streamlit
pypdf
sentence-transformers
faiss-cpu
numpy
torch
transformers
scikit-learn
