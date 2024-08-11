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
###Upload a PDF
Click on "Upload a PDF file" to select your document.
The app will process the PDF, displaying a progress bar during the extraction.

###Ask Questions
After the PDF is processed, you can input your question in the text box provided.
The app will retrieve relevant information and generate a concise answer using the gemma2:2b model.
