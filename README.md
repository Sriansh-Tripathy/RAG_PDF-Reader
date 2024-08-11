# RAG_PDF-Reader

This repository contains a Streamlit-based application that implements a Retrieval-Augmented Generation (RAG) system using the `gemma2:2b` model from Ollama. The application allows users to upload a PDF, process its contents, and interact with the document by asking questions.

## Features

### PDF Upload
Upload and extract text from PDF files.

### Text Chunking
Automatically split the extracted text into chunks for efficient processing.

### RAG with gemma2:2b
Utilize the `gemma2:2b` model for generating concise answers based on the retrieved context from the PDF.

### Interactive Interface
User-friendly Streamlit interface with real-time progress tracking.

## Installation

### Clone the Repository
```bash
git clone https://github.com/Sriansh-Tripathy/RAG_PDF-Reader.git
````
### Install Requirements
```bash
pip install -r requirements.txt
````
### Running app
```bash
streamlit run app.py
````
### Upload a PDF

Click on "Upload a PDF file" to select your document.
The app will process the PDF, displaying a progress bar during the extraction.

### Ask Questions
After the PDF is processed, you can input your question in the text box provided.
The app will retrieve relevant information and generate a concise answer using the gemma2:2b model.

## Code Explanation
This section provides a detailed walkthrough of the key components and functionality of the code in this repository.

### 1. Importing Libraries
The code begins by importing essential libraries:
```bash
import streamlit as st
import torch
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
import PyPDF2
import time
````
Streamlit: Used to create the web interface.

torch: Though imported, it is not used directly in the code. It might be necessary for the Ollama model, which utilizes PyTorch.

ollama: Used to interact with the gemma2:2b model for generating answers.

langchain.text_splitter: Helps split the extracted text into manageable chunks.

langchain_community.vectorstores.FAISS: Used to build a vector store for efficient retrieval.

langchain_community.embeddings.OllamaEmbeddings: Generates embeddings for the text chunks using the gemma2:2b model.

PyPDF2: Used to extract text from the uploaded PDF files.

time: Used for simulating delays and updating the UI.
### 2. Loading and Processing the PDF
The load_pdf function reads the uploaded PDF and extracts the text:
```bash
def load_pdf(file):
    reader = PyPDF2.PdfReader(file)
    total_pages = len(reader.pages)
    text = ""

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, page in enumerate(reader.pages):
        text += page.extract_text()
        progress_bar.progress((i + 1) / total_pages)
        status_text.text(f"Reading page {i + 1} of {total_pages}")
        time.sleep(0.05)

    progress_bar.empty()
    status_text.text("PDF loading complete.")
    return text
````
### 3. Splitting Text into Chunks
The split_pdf_text function splits the extracted text into smaller chunks:
```bash
def split_pdf_text(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_texts = text_splitter.split_text(pdf_text)
    return [Document(page_content=chunk) for chunk in split_texts]
````
Text Chunking: The text is divided into chunks of 1000 characters with a 200-character overlap to maintain context between chunks.

Document Objects: Each chunk is stored in a Document object.
### 4. Creating a Retriever
The create_retriever function builds a retriever using FAISS and the Ollama embeddings:
```bash
def create_retriever(documents):
    embeddings = OllamaEmbeddings(model="gemma2:2b")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever()

````
Ollama Embeddings: Generates embeddings for the text chunks using the gemma2:2b model.

FAISS Vectorstore: Indexes these embeddings for efficient retrieval based on user queries.
### 5. Generating Answers with RAG
The rag_chain function performs the core RAG operation:
```bash
def rag_chain(question, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return ollama_llm(question, formatted_context)
````
Retrieving Context: The retriever fetches relevant chunks from the vector store.

Generating Responses: The ollama_llm function is called to generate a concise answer based on the retrieved context.
### 6. Streamlit Interface
Finally, the Streamlit app ties everything together:
```bash
st.title("RAG with gemma2:2b")
st.write("Ask questions about the provided context")

if 'documents' not in st.session_state:
    st.session_state.documents = []
    st.session_state.retriever = None

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        pdf_text = load_pdf(uploaded_file)
        st.session_state.documents = split_pdf_text(pdf_text)
        st.session_state.retriever = create_retriever(st.session_state.documents)

    question = st.text_input("Enter your question here...")

    if question:
        with st.spinner("Retrieving answer..."):
            answer = rag_chain(question, st.session_state.retriever)
            st.write("Answer:", answer)
````
Title and Description: The app displays a title and a brief description.

Session State: Documents and retrievers are stored in the session state to avoid redundant processing.

PDF Upload: Users can upload a PDF, which is processed, and its contents are indexed.

Question Input: Users can input questions, which are answered based on the retrieved context.
