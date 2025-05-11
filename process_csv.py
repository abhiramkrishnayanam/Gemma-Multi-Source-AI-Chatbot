import os
import pandas as pd
import streamlit as st
import tempfile  # <-- Add this
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def process_csv(uploaded_file):
    # Read file content safely
    csv_text = uploaded_file.read().decode("utf-8", errors="replace")

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv", encoding="utf-8") as tmp:
        tmp.write(csv_text)
        tmp_path = tmp.name

    # Reload the data for dataframe
    df = pd.read_csv(tmp_path)

    # Load and process text
    loader = TextLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(final_docs, embeddings)

    return vector_store, df.shape
