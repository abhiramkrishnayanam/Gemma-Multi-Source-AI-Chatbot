import os
import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq
from process_csv import process_csv
from process_pdf import process_pdf
from process_url import process_website
from process_youtube import process_youtube_transcript
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import time

# Set API keys
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize Groq llm
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Custom CSS
st.markdown("""
    <style>
    .center-heading {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subheading {
        text-align: center;
        font-size: 16px;
        color: #cccccc;
        margin-bottom: 20px;
    }

    .welcome-message {
        text-align: center;
        font-size: 18px;
        margin-bottom: 10px;
    }

    .stTextInput>div>div>input,
    .stTextArea>div>textarea,
    .stSelectbox>div>div>div {
        font-size: 14px;
        height: 35px;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 800px;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Display heading and caption
st.markdown('<div class="welcome-message">👋 Welcome to your AI-powered assistant!</div>', unsafe_allow_html=True)
st.markdown('<div class="center-heading">📊📄 Gemma Multi-Source AI Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subheading">Ask questions from <b>CSV, PDF, YouTube</b>, or <b>Website</b> using the <b>Gemma2-9b-it</b> model running on <b>Groq</b> for ultra-fast inference.</div>', unsafe_allow_html=True)



# selector for both file types
file_type = st.selectbox("Select File Type", ["Select", "CSV", "PDF", "Website URL", "Youtube"])

uploaded_file = None
video_url = None
web_url = None

if file_type != "Select":
    if file_type in ["CSV", "PDF"]:
        uploaded_file = st.file_uploader(f"📤 Upload your {file_type} file here", type=[file_type.lower()])
    elif file_type == "Youtube":
        video_url = st.text_input("📺 Enter the YouTube video URL")
        st.write("You entered:", video_url)
    elif file_type == "Website URL":
        web_url = st.text_input("📺 Enter the Website URL")
        st.write("You entered:", web_url)

    # Process the uploaded or entered input
    if file_type == "CSV" and uploaded_file:
        if uploaded_file:
            # Process csv fie
            st.write("Processing your CSV file...")

            # Reset pointer before reading inside process_csv
            uploaded_file.seek(0)
            vector_store, df_shape = process_csv(uploaded_file)

            st.session_state.vectors = vector_store
            st.success("✅ CSV Vector store created successfully!")
            st.session_state.num_rows = df_shape[0]
            st.session_state.num_columns = df_shape[1]

            # Reset pointer before re-reading for preview
            uploaded_file.seek(0)
            st.write("📝 Document Preview:")
            st.dataframe(pd.read_csv(uploaded_file).head())
        else:
            st.info("📁 Please upload your CSV file to continue.")

    elif file_type == "PDF":
        if uploaded_file:
            st.write("Processing your PDF file...")
            vector_store = process_pdf(uploaded_file)
            st.session_state.vectors = vector_store
            st.success("✅ PDF Vector store created successfully!")
        else:
            st.info("📁 Please upload your PDF file to continue.")

    elif file_type == "Youtube" and video_url:
        st.write("Processing the YouTube video...")
        vector_store, transcript_text = process_youtube_transcript(video_url)
        if vector_store:
            st.session_state.vectors = vector_store
            st.session_state.transcript = transcript_text
            st.success("✅ YouTube Vector store created successfully!")
            with st.expander("📄 Transcript Preview"):
                st.write(transcript_text[:1000] + "..." if transcript_text else "No transcript available.")

    elif file_type == "Website URL" and web_url:
        st.write("Processing the Website content...")
        vector_store, summary = process_website(web_url)
        if vector_store:
            st.session_state.vectors = vector_store
            st.session_state.website_summary = summary  # Optional: store snippet for display
            st.success("✅ Website URL Vector store created successfully!")
            st.write("Website content snippet:")
            st.write(summary)



    elif file_type in ["CSV", "PDF"] and not uploaded_file:
        st.info(f"📁 Please upload your selected {file_type} file to continue.")

# Ask question based on the content
question = st.text_input("💬 Ask a question based on the uploaded content")

if question and "vectors" in st.session_state:

    prompt_template = None  # Initialize

    if file_type == "CSV":
        prompt_template = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the question based on the CSV data provided below.

        - If the user asks for number of rows, respond: {num_rows}
        - If the user asks for number of columns, respond: {num_columns}

        <context>
        {context}
        </context>

        Question: {input}
        """)

        # Replace dynamic parts using partials
        prompt_template = prompt_template.partial(
            num_rows=str(st.session_state.get("num_rows", "unknown")),
            num_columns=str(st.session_state.get("num_columns", "unknown"))
        )

    elif file_type == "PDF":
        prompt_template = ChatPromptTemplate.from_template("""
        Answer the question based on the provided content only.
        Please provide the most accurate response based on the question.

        <context>
        {context}
        </context>

        Question: {input}
        """)

    elif file_type == "Youtube":
        # create prompt template for youtube video content
        prompt_template = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the question based on the transcript of the YouTube video.
        Please provide the most accurate response based on the question.

        <context>
        {context}
        </context>

        Question: {input}
        """)

    elif file_type == "Website URL":
        # Create prompt template for website URL content
        prompt_template = ChatPromptTemplate.from_template("""
            You are a helpful assistant. Answer the question based only on the content extracted from the given website URL.
            Do not make assumptions. If the information is not found in the provided context, say "I couldn't find that information on the page."

            <context>
            {context}
            </context>

            Question: {input}
            """)

    # Now proceed only if prompt_template was defined
    if prompt_template is None:
        st.error("❌ Unsupported file type or prompt could not be created.")
    else:
        try:
            # Create document chain and retrieval chain
            doc_chain = create_stuff_documents_chain(llm, prompt_template)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, doc_chain)

            # Run the chain
            start = time.process_time()
            response = retrieval_chain.invoke({'input': question})
            end = time.process_time()

            # Display results
            st.write("🧠 **Answer:**", response.get('answer', 'No answer returned.'))
            st.caption(f"⏱️ Response Time: {end - start:.2f}s")

            with st.expander("🔍 Relevant Document Chunks"):
                for i, doc in enumerate(response.get("context", [])):
                    st.markdown(f"**Chunk {i + 1}:**")
                    st.code(doc.page_content)

        except Exception as e:
            st.error(f"⚠️ An error occurred during retrieval: {str(e)}")

