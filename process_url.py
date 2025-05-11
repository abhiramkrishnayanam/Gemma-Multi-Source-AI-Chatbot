import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Function to fetch and parse the website content
def fetch_website_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Will raise an exception for bad HTTP status codes
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract all text from the page
        text = soup.get_text()

        #Clean up text (you can do additional cleaning if needed)
        text = " ".join(text.split())

        return text
    except Exception as e:
        raise ValueError(f"Error fetching website content: {str(e)}")

# Function to process website URL and create vector store
def process_website(url):
    # Fetch website content
    website_text = fetch_website_content(url)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Wrap the website text in a Document object
    document = Document(page_content=website_text)
    final_documents = text_splitter.split_documents([document])

    #Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model ="models/embedding-001")
    vector_store = FAISS.from_documents(final_documents, embeddings)

    return vector_store, website_text[:500]
