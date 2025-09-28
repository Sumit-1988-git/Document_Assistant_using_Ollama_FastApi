from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi import HTTPException
import os
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import numpy as np
from langchain_core.documents import Document

app = FastAPI()

# Define the Ollama model
model_name = "llama3.2:1b"

# Initialize the OllamaEmbeddings object
embedding = OllamaEmbeddings(model=model_name)

# Set up the TF-IDF Vectorizer for keyword-based search
tfidf_vectorizer = TfidfVectorizer()

# Initialize FAISS index
faiss_index = None

# Initialize corpus and documents list
corpus = []
documents_list = []  # Store Document objects for hybrid search

# Upload PDF files and process them
@app.post("/upload_pdf")
async def upload_pdf(files: list[UploadFile] = File(...)):
    global faiss_index, documents_list  # Make sure the FAISS index is updated globally
    documents = []

    # Ensure the 'temp/' directory exists
    os.makedirs("temp", exist_ok=True)

    # Process each uploaded file
    for file in files:
        # Save the uploaded file temporarily
        file_location = f"temp/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Load and process the PDF
        loader = PyPDFLoader(file_location)
        docs = loader.load()
        if not docs:
            print(f"Warning: No content found in {file_location}")
        documents.extend(docs)

        # Delete the file after processing
        os.remove(file_location)

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks generated after text splitting.")
    
    # Check corpus before fitting TF-IDF
    global corpus
    corpus = [doc.page_content for doc in chunks]
    documents_list = chunks  # Store the Document objects
    
    if not corpus:
        raise HTTPException(status_code=400, detail="No valid content to fit TF-IDF.")
    
    # Fit the TF-IDF vectorizer on the newly uploaded documents
    tfidf_vectorizer.fit(corpus)

    # Update FAISS index with new chunks
    faiss_index = FAISS.from_documents(chunks, embedding)
    
    # print(corpus)
   
    return {"message": "PDFs uploaded and processed successfully!"}

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Document Assistant</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #5F2C82, #49A09D);
            color: #fff;
            margin: 0;
            padding: 0;
        }

        header {
            background-color: #1A1A2E;
            padding: 20px;
            text-align: center;
        }

        header h1 {
            font-size: 3em;
            margin: 0;
            color: #fff;
        }

        .container {
            width: 80%;
            margin: 40px auto;
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            color: #333;
        }

        h2 {
            font-size: 1.5em;
            color: #333;
        }

        ul {
            list-style: none;
            padding: 0;
        }

        li {
            margin: 10px 0;
        }

        .question-form {
            margin-top: 20px;
        }

        label, input {
            display: block;
            width: 100%;
            margin-bottom: 10px;
        }

        input[type="text"] {
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 1.1em;
        }

        button {
            background-color: #49A09D;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 1.1em;
            cursor: pointer;
        }

        button:hover {
            background-color: #3e8e8c;
        }

        footer {
            text-align: center;
            margin-top: 40px;
            color: #fff;
        }

        /* Spinner styles */
        #spinner {
            display: none;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            border: 4px solid #f3f3f3;
            border-top: 4px solid #49A09D;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 2s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Hide success message initially */
        #success-message {
            display: none;
            color: green;
        }

        /* Hide the question form initially */
        #ask-question {
            display: none;
            margin-top: 20px;
        }

        #response {
            margin-top: 20px;
            font-size: 1.2em;
            color: #333;
        }
    </style>
</head>
<body>

<header>
    <h1>Interactive Document Assistant</h1>
</header>

<div class="container">
    <h2>Instructions:</h2>
    <ul>
    <li><b>1. Click on the "Upload PDF" button to upload your PDF files.</b></li>
    <li><b>2. The system will process the PDFs and split them into smaller chunks.</b></li>
    <li><b>3. Once the PDF is processed, you will see a success message and be able to ask questions.</b></li>
    <li><b>4. Type your question in the "Ask a Question" field and click "Submit Question".</b></li>
    <li><b>5. The assistant will respond with an answer based on the uploaded PDFs.</b></li>
    <br>
    <li><b><u>Note:</u></b></li>
    <li><b>It may take time to load PDFs based on the number, size and contents</b></li>
    </ul>

    <!-- Upload form -->
    <form id="upload-form" action="/upload_pdf" method="post" enctype="multipart/form-data">
        <label for="files">Upload PDF files:</label>
        <input type="file" id="files" name="files" multiple required>
        <button type="submit">Upload PDF</button>
    </form>

    <div id="success-message" style="display: none; color: green;">
        <p>PDFs uploaded and processed successfully!</p>
    </div>

    <!-- New interface after successful upload -->
    <div id="ask-question" style="display: none; margin-top: 20px;">
        <h2>Ask a Question:</h2>
        <form id="ask-form">
            <label for="question">Your Question:</label>
            <input type="text" id="question" name="question" required>
            <button type="submit">Submit Question</button>
        </form>

        <div id="response"></div>
    </div>
</div>

<!-- Spinner -->
<div id="spinner"></div>

<footer>
    <p>Powered by Interactive Assistant</p>
</footer>

<script>
    document.getElementById('upload-form').addEventListener('submit', function(event) {
        event.preventDefault();
        let formData = new FormData(this);

        document.getElementById('spinner').style.display = 'block';

        fetch('/upload_pdf', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                document.getElementById('spinner').style.display = 'none';
                document.getElementById('success-message').style.display = 'block';
                document.getElementById('ask-question').style.display = 'block';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('spinner').style.display = 'none';
        });
    });

    // Handle question submission via AJAX
    document.getElementById('ask-form').addEventListener('submit', function(event) {
        event.preventDefault();
        let formData = new FormData(this);

        document.getElementById('spinner').style.display = 'block';

        fetch('/ask', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(data => {
            document.getElementById('spinner').style.display = 'none';
            document.getElementById('response').innerHTML = data;  // Display the answer from FastAPI
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('spinner').style.display = 'none';
        });
    });
</script>

</body>
</html>
"""

def ensure_document(obj):
    """Ensure the object is a Document, convert if necessary"""
    if isinstance(obj, Document):
        return obj
    elif hasattr(obj, 'page_content'):
        # If it has page_content but isn't a Document, create one
        return Document(page_content=obj.page_content)
    else:
        # Convert string or any other object to Document
        return Document(page_content=str(obj))

def get_hybrid_results(user_query, faiss_index, tfidf_vectorizer, corpus, documents_list, top_k=5):
    """Simplified hybrid search that ensures Document objects"""
    try:
        print("Starting hybrid search...")
        
        # Get FAISS results
        faiss_results = faiss_index.similarity_search(user_query, k=top_k)
        print(f"FAISS returned {len(faiss_results)} results")
        
        # Convert all FAISS results to Document objects
        faiss_docs = [ensure_document(doc) for doc in faiss_results]
        
        # If we don't have a proper corpus or documents_list, just return FAISS results
        if not corpus or not documents_list or len(corpus) != len(documents_list):
            print("Corpus/documents mismatch, returning FAISS results only")
            return faiss_docs[:top_k]
        
        # Get TF-IDF results
        try:
            query_vec = tfidf_vectorizer.transform([user_query])
            corpus_vec = tfidf_vectorizer.transform(corpus)
            similarities = (query_vec * corpus_vec.T).toarray().flatten()
            
            # Get top TF-IDF indices
            top_tfidf_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Convert TF-IDF results to Document objects
            tfidf_docs = []
            for idx in top_tfidf_indices:
                if idx < len(documents_list) and similarities[idx] > 0:
                    doc = ensure_document(documents_list[idx])
                    tfidf_docs.append(doc)
            
            print(f"TF-IDF returned {len(tfidf_docs)} results")
            
            # Combine results (simple deduplication)
            combined_docs = []
            seen_content = set()
            
            # Add TF-IDF results first
            for doc in tfidf_docs:
                content_hash = hash(doc.page_content[:100])  # Hash first 100 chars for dedup
                if content_hash not in seen_content:
                    combined_docs.append(doc)
                    seen_content.add(content_hash)
            
            # Add FAISS results that aren't duplicates
            for doc in faiss_docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    combined_docs.append(doc)
                    seen_content.add(content_hash)
            
            return combined_docs[:top_k]
            
        except Exception as tfidf_error:
            print(f"TF-IDF search failed: {tfidf_error}, using FAISS only")
            return faiss_docs[:top_k]
            
    except Exception as e:
        print(f"Hybrid search failed completely: {e}")
        # Final fallback - simple FAISS search
        try:
            simple_results = faiss_index.similarity_search(user_query, k=top_k)
            return [ensure_document(doc) for doc in simple_results]
        except:
            return []

@app.post("/ask")
async def ask_question(request: Request):
    form_data = await request.form()
    user_query = form_data.get("question")

    if not user_query:
        return HTMLResponse(content="<p>Please enter a question.</p><a href='/'>Back to Home</a>")
    
    if faiss_index is None:
        return HTMLResponse(content="<p>Please upload PDFs first before asking questions.</p><a href='/'>Back to Home</a>")

    try:
        # SIMPLE APPROACH: Use only FAISS retrieval chain (most reliable)
        print("Using simple FAISS retrieval approach...")
        
        # Create the prompt template
        prompt_template = ChatPromptTemplate.from_template("""
        You are a helpful assistant that provides accurate and specific information based EXCLUSIVELY on the provided document context.
        
        DOCUMENT CONTEXT:
        {context}
        
        USER QUESTION: {input}
        
        INSTRUCTIONS:
        1. Answer the question using ONLY the information from the document context above.
        2. Be specific and cite relevant details from the context.
        3. Do not make up information or use external knowledge.
        4. If the question is ambiguous, ask for clarification but first try to answer based on the context.
        
        ANSWER:
        """)    

        # Create the OllamaLLM object
        llm = OllamaLLM(model=model_name, temperature=0.1)

        # Use the standard retrieval chain (most reliable)
        retriever = faiss_index.as_retriever(search_kwargs={"k": 4})
        retrieval_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt_template))
        
        response = retrieval_chain.invoke({"input": user_query})
        answer = response["answer"]

    except Exception as e:
        print(f"Error in question processing: {e}")
        answer = "I encountered an error while processing your question. Please try uploading the PDFs again or rephrasing your question."

    return HTMLResponse(content=f"""
    <html>
       <body>
            <div class="answer">
                <strong>Question:</strong> {user_query}<br><br>
                <strong>Assistant's Answer:</strong> {answer}
            </div>
            <br>
            <a href="/">← Back to Home</a>
        </body>
    </html>
    """)

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector (list or numpy array)
        vec2: Second vector (list or numpy array)
    
    Returns:
        float: Cosine similarity score
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Ensure vectors are 1D
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)