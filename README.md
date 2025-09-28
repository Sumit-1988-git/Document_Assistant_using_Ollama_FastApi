# Interactive Document Assistant using Ollama and FastAPI

---

This project is a FastAPI-based web application that allows users to upload PDF files, which are then processed into smaller chunks and indexed for efficient search. Users can then ask questions based on the contents of the uploaded PDFs, and the assistant will provide answers by retrieving relevant information from the indexed documents using hybrid search with FAISS and TF-IDF.

## Features

- Upload multiple PDF files.
- Process and split PDF content into smaller chunks.
- Index the content with FAISS for efficient similarity search.
- Use TF-IDF for keyword-based search to enhance retrieval.
- Hybrid search approach combining FAISS and TF-IDF.
- Simple and intuitive web interface to interact with the assistant.
-  **Customizable model selection**: Users can choose their own model for embeddings and answering queries.


## Requirements

To run this project, you need to have Python 3.8 or later installed along with the following libraries:

- FastAPI
- langchain
- scikit-learn
- numpy
- PyPDF2
- FAISS
- Ollama
- Uvicorn (for running the FastAPI app)

You can install the dependencies using the following command:

```
pip install -r requirements.txt

```

## Setup

**1. Clone the repository:**
```
git clone https://github.com/your-username/interactive-document-assistant.git
```

**2. Navigate to the project directory:**
```
cd interactive-document-assistant

```

**3. Install the required dependencies:**
```
pip install -r requirements.txt
```

**4. Run the FastAPI app:**
```
uvicorn main:app --reload

```

**5. Open Application**
```
Open your browser and go to http://127.0.0.1:8000 to use the application.
```

## Usage

---

**1. Upload PDF Files**

* Click on the "Upload PDF" button to select and upload your PDF files.

* The system will process and split the PDFs into smaller chunks. Once the processing is complete, you will be able to ask questions.

**2. Ask a Question**

* After uploading the PDFs, you can enter your question in the "Ask a Question" section.

* The assistant will provide answers based on the content from the uploaded PDFs.

**3. Instructions**

* The system processes PDFs and splits them into smaller chunks for efficient search.

* You can ask questions, and the assistant will respond based on the documents uploaded.

* Please note that processing large PDFs might take some time, depending on the number of files and their size.

## Code Overview

---

**1. main.py**

This file contains the FastAPI app with the following main components:

* **PDF Upload Endpoint (/upload_pdf):** This endpoint handles the upload of PDF files and processes them into smaller chunks using the PyPDFLoader and RecursiveCharacterTextSplitter.

* **Question Endpoint (/ask):** This endpoint processes user questions by performing a hybrid search using both FAISS and TF-IDF vectorization to find relevant content from the uploaded documents.

* **Home Page (/):** A simple HTML page where users can upload PDFs and ask questions.

**2. FAISS and TF-IDF**

* **FAISS Index:** FAISS is used to create an index for fast similarity search. It helps find documents that are similar to the user's query.

* **TF-IDF Vectorizer:** TF-IDF is used to enhance the search by considering the importance of terms in the documents.

## Customization

* You can change the model used for embeddings by modifying the model_name variable in main.py.

* You can adjust the chunk size and overlap in the RecursiveCharacterTextSplitter to suit your needs.

* The assistant's response behavior can be modified in the ChatPromptTemplate.

## Troubleshooting

* **Error: "No content found in the uploaded file"**

  This warning appears if the uploaded PDF file doesn't contain any text content that can be extracted.

* **Error: "No valid content to fit TF-IDF."**

  This error occurs if no valid text content is found to process after splitting the document.

* **Error: "Corpus/documents mismatch."**

  This indicates that the corpus (list of document contents) and the document objects do not align, which could happen if there are issues during document splitting.




