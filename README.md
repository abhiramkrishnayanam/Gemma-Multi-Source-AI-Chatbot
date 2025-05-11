# 📊🧠 Gemma Multi-Source AI Chatbot

Welcome to the **Gemma Multi-Source AI Chatbot** – a powerful Streamlit-based web app that leverages the **Gemma 2 9B IT model** hosted on **Groq** to enable question-answering from multiple content sources including **CSV files, PDFs, YouTube videos, and websites**.

---

## 📌 Overview

This chatbot application allows users to upload or input various data sources and receive intelligent, context-aware responses powered by the **Gemma large language model**. Whether you're analyzing structured data, reading a PDF, exploring video transcripts, or parsing website content, this tool provides an intuitive interface for seamless interaction.

---

## ✨ Features

- 📁 **CSV Upload**: Analyze and ask questions from structured tabular data.
- 📄 **PDF Upload**: Extract and understand textual content from PDFs.
- 📺 **YouTube Integration**: Process and query video transcripts directly from URLs.
- 🌐 **Website Scraping**: Summarize and chat with content from public web pages.
- ⚙️ **Groq API Integration**: Ultra-fast response generation with low latency.
- 💬 **Session-Based Context Handling**: Smooth conversational experience across sources.

---

## 🧰 Technologies Used

- **Streamlit** – UI/UX framework
- **Gemma 2 9B IT Model** – Language model used for inference
- **Groq API** – For lightning-fast inference
- **LangChain** – For text processing and vector search
- **FAISS / Chroma** – Vector store for document embeddings
- **PyMuPDF / pdfplumber** – For PDF text extraction
- **BeautifulSoup** – For website content scraping
- **YoutubeTranscriptAPI** – For transcript extraction from videos

---

## 🚀 Getting Started

To run this project on your local system:

1. Clone the repository
2. Install required dependencies from `requirements.txt`
3. Set up API keys (Groq and others)
4. Launch the app using Streamlit

> This application is modular and easy to extend — you can add more data sources or swap the LLM backend with minimal changes.

---

## 🧪 Use Cases

- Academic research
- Data analysis and interpretation
- Content summarization
- Conversational retrieval from diverse sources
- Educational tool for learning and exploration

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo, open issues, or submit pull requests.

---

## 🙏 Acknowledgments

- The Gemma model by Google DeepMind
- Groq for blazing-fast inference platform
- OpenAI, LangChain, and the open-source community

---

> 🔔 *For any inquiries or suggestions, feel free to open an issue or contact the maintainer via GitHub.*

