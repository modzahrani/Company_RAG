# Company RAG

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about company data and receive answers grounded in a private knowledge base.

The project uses **LangChain** for the RAG pipeline, **ChromaDB** for vector storage, **Hugging Face embeddings** for semantic search, and **Google Gemini** for response generation.

## Features

* Semantic search over company documents
* Retrieval-Augmented Generation (RAG)
* ChromaDB vector database
* Hugging Face `all-MiniLM-L6-v2` embeddings
* Google Gemini for answer generation
* Markdown-based knowledge base
* Gradio interface
* Environment variable support for API keys
* Persistent local vector database

## Architecture

```text
User Question
     │
     ▼
Embedding Model
     │
     ▼
ChromaDB
     │
     ▼
Relevant Documents
     │
     ▼
Context + Question
     │
     ▼
Google Gemini
     │
     ▼
Generated Answer
```

## Tech Stack

* **Python**
* **LangChain**
* **ChromaDB**
* **Hugging Face Sentence Transformers**
* **Google Gemini**
* **Gradio**
* **uv** for Python package and environment management

## Project Structure

```text
Company_RAG/
│
├── knowledge_base/       # Company documents used for retrieval
│
├── vector_db/            # Local ChromaDB database
│
├── main.py               # Gradio application
├── rag_pipeline.py       # RAG pipeline and retrieval logic
│
├── .env                  # API keys and environment variables
├── .gitignore
├── pyproject.toml        # Project dependencies and configuration
└── uv.lock               # Locked dependency versions
```

> `vector_db/` and `.env` are excluded from Git because they contain generated/local data and secrets.

## Knowledge Base

The knowledge base contains Markdown documents representing different types of company information, including:

* Company information
* Careers and culture
* Employees
* Products
* Contracts
* Other internal company data

These documents are processed and converted into vector embeddings before being stored in ChromaDB.

## RAG Pipeline

The application follows a standard RAG workflow:

1. Load documents from the knowledge base.
2. Split documents into smaller chunks.
3. Generate embeddings using `all-MiniLM-L6-v2`.
4. Store embeddings in ChromaDB.
5. Convert the user's question into an embedding.
6. Retrieve the most relevant document chunks.
7. Provide the retrieved context to Gemini.
8. Generate an answer grounded in the retrieved information.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/modzahrani/Company_RAG.git
cd Company_RAG
```

### 2. Install dependencies

This project uses `uv`.

```bash
uv sync
```

### 3. Activate the virtual environment

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

Replace the value with your Google Gemini API key.

## Running the Application

After installing the dependencies and configuring the environment:

```bash
python main.py
```

The Gradio interface will start locally.

## Retrieval Experiments

The project can also be used to experiment with different retrieval strategies and parameters.

For example, Chroma can be configured to use similarity search or Maximum Marginal Relevance (MMR):

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)
```

### MMR Parameters

* `k` — Number of final chunks returned.
* `fetch_k` — Number of candidate chunks considered before MMR selection.
* `lambda_mult` — Controls the balance between relevance and diversity.

The project can be evaluated using retrieval metrics such as:

* **Hit@K**
* **Recall@K**
* **Context Precision**
* **Context Recall**
* **Faithfulness**
* **Answer Relevancy**

## Future Improvements

* Add automated RAG evaluation with Ragas
* Compare similarity search against MMR
* Experiment with different chunk sizes and overlap
* Add reranking
* Improve query transformation
* Add conversation history
* Add metadata filtering
* Improve retrieval evaluation with a labeled test dataset
* Deploy the application

## Author

**Mohammed Alzahrani**

Software Engineering Graduate
GitHub: [@modzahrani](https://github.com/modzahrani)
