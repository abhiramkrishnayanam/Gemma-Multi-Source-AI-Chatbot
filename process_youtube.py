from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import os
# from youtube_transcript_api.formatters import TextFormatter
# from youtube_transcript_api._api import TranscriptList
from urllib.parse import urlparse, parse_qs

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def process_youtube_transcript(video_url):
    try:
        # Step 1: Extract video ID
        query = urlparse(video_url).query
        video_id = parse_qs(query).get("v")
        if not video_id:
            # Handle shortened youtu.be links
            parsed_path = urlparse(video_url).path
            video_id = parsed_path.strip("/")
        else:
            video_id = video_id[0]

        # Step 2: Fetch transcript using the correct method
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])  # Fetch English transcript
        text = " ".join([item['text'] for item in transcript])

        #Step 3: Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [Document(page_content=text)]
        final_docs = splitter.split_documents(docs)

        #step 4 : embed
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(final_docs, embeddings)

        return vector_store, text

    except Exception as e:
        raise Exception(f"Error processing YouTube transcript: {str(e)}")

# if __name__ == "__main__":
#     test_url = "https://youtu.be/ldFONBo2CR0?si=t-0OzXPHQiP92-WN"  # Replace with any valid video URL that has a transcript
#     try:
#         result_vector_store = process_youtube_transcript(test_url)
#         print("✅ Transcript processed and embedded successfully.")
#         # Count number of documents using index
#         doc_count = result_vector_store.index.ntotal  # ✅ Correct way to count chunks
#         print(f"Number of chunks created: {doc_count}")
#     except Exception as e:
#         print(f"❌ Error: {e}")


