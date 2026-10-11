# Corporate RAG Assistant

A corporate knowledge assistant that answers questions using information retrieved from PDF documents.

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) application in Python using LangChain, OpenAI, Chroma, and Streamlit.

## Project Goal

The goal of this project is to understand and implement the main components of a RAG system:

* PDF document ingestion and validation
* Text splitting and chunk management
* Text embeddings and vector storage
* Semantic similarity search
* Context-aware answer generation
* A chat interface for interacting with corporate documents
* Basic security, logging, and error handling

The application is designed as a learning project with a modular architecture that can be extended in the future.

## Features

* Load and process PDF documents
* Validate PDF file extensions, signatures, and file sizes
* Split documents into overlapping text chunks
* Generate embeddings using OpenAI
* Store document vectors in a persistent Chroma database
* Retrieve relevant document chunks using semantic search
* Generate answers using a retrieval-augmented prompt
* Display source documents in the Streamlit interface
* Handle empty questions and missing retrieval results
* Configure the application through environment variables
* Delete indexed documents and rebuild the vector index
* Log application errors without displaying technical exception details to users
* Run automated tests without making paid OpenAI API requests

## Architecture

The application follows this workflow:

1. **Document ingestion** — PDF files are loaded and validated.
2. **Chunking** — document text is split into smaller overlapping chunks.
3. **Embedding generation** — each chunk is converted into a vector representation.
4. **Vector storage** — embeddings and document metadata are stored in Chroma.
5. **Retrieval** — the most relevant chunks are retrieved for a user question.
6. **Prompt construction** — the retrieved context is added to the RAG prompt.
7. **Answer generation** — the language model generates a response based on the provided context.
8. **User interface** — Streamlit displays the answer and its source information.

## Technology Stack

* **Python** — application development
* **LangChain** — RAG pipeline and model integration
* **OpenAI API** — text embeddings and answer generation
* **Chroma** — persistent vector database
* **PyPDF** — PDF document loading
* **Streamlit** — interactive chat interface
* **Pydantic Settings** — configuration management
* **pytest** — automated testing

## Project Structure

```text
corporate-rag-assistant/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── pdfs/
│   └── chroma/
├── scripts/
│   ├── index_documents.py
│   ├── reindex_documents.py
│   └── delete_document.py
├── src/
│   └── corporate_rag/
│       ├── config.py
│       ├── ingestion.py
│       ├── chunking.py
│       ├── embeddings.py
│       ├── indexing.py
│       ├── vector_store.py
│       ├── retrieval.py
│       ├── document_management.py
│       ├── llm.py
│       ├── rag.py
│       ├── logging_config.py
│       └── app.py
├── tests/
|    ├── test_vector_store.py
|    ├── test_rag.py
|    ├── test_app.py
|    ├── test_model_factories.py
|    └── test_document_management.py
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── pyproject.toml
└── README.md
```

The `src/corporate_rag/` package contains the application logic. The `scripts/` directory contains command-line utilities for indexing and managing documents. PDF files are stored separately from the application source code.

## Requirements

* Python 3.13 or a compatible Python version supported by the installed dependencies
* An OpenAI API key with API access enabled
* OpenAI API billing configured for actual API usage

The application can be installed and its unit tests can be run without making paid API requests. Actual embedding generation and answer generation require working API access.

## Installation

Clone the repository and open its root directory:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd corporate-rag-assistant
```

Create and activate a virtual environment.

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
```

Install the project and its dependencies:

```bash
python -m pip install -e .
```

## Configuration

Create a `.env` file in the project root using `.env.example` as a template.

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

Replace the placeholder with your own API key.

**Security:** Never commit `.env` or expose your API key in source code, logs, screenshots, or public repositories.

The application uses Pydantic Settings to load configuration. Model names, chunk sizes, retrieval settings, and data paths are configured in `src/corporate_rag/config.py`.

## Adding Documents

Place your PDF documents in:

```text
data/pdfs/
```

The application loads PDF files from this directory. Only documents intended for the application's knowledge base should be added.

## Indexing Documents

Build the vector index from the available PDFs:

```bash
python scripts/index_documents.py
```

The script loads and validates the documents, splits them into chunks, generates embeddings, stores the chunks in Chroma, and runs a sample similarity search.

The first indexing run requires working OpenAI API access.

## 🐳 Running with Docker

The application can also run inside a Docker container, providing an isolated and reproducible environment without requiring a local Python installation.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- An OpenAI API key with API access and billing configured

### Build the Docker image

Run the following command from the project root directory, where the `Dockerfile` is located:

```bash
docker build -t corporate-rag-assistant .
```

### Run the container

```bash
docker run --rm -p 8501:8501 --env-file .env corporate-rag-assistant
```

The application will be available at:

http://localhost:8501

### Notes

- The `.env` file is used to provide configuration values and the OpenAI API key to the container.
- Keep `.env` out of version control. Never commit API keys or other secrets to GitHub.
- The `--rm` option automatically removes the container when it stops.
- The `-p 8501:8501` option maps the container's Streamlit port to the host machine.
- Ensure that any required persistent data, such as uploaded PDFs or the Chroma vector store, is stored in a Docker volume or mounted directory if it needs to survive container removal.

For development, the application can also be run locally using Streamlit, as described in the section below.

## Running the Application

Start Streamlit:

```bash
python -m streamlit run src/corporate_rag/app.py
```

Open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

Ask questions about the indexed documents. The assistant retrieves relevant context and generates an answer based on that context.

## Document Management

### Rebuild the index

```bash
python scripts/reindex_documents.py
```

This removes the existing indexed chunks and builds a new index from the PDF files in `data/pdfs/`.

**Warning:** The operation clears the existing index before rebuilding it. Keep backups if the indexed data is important.

### Delete a document from the index

```bash
python scripts/delete_document.py --source "EXACT_SOURCE_PATH"
```

Use the exact source path stored in the document metadata.

This command removes the document's chunks from the vector store. It does not delete the original PDF file.

## Testing

Run the automated tests:

```bash
pytest -q
```

The tests cover core functionality such as document processing, chunking, retrieval, RAG behavior, configuration of model clients, and document management.

## Future Improvements

Possible extensions include:

* User authentication and document-level access control
* Support for additional document formats
* Better retrieval evaluation and relevance metrics
* Hybrid search and reranking
* Automated evaluation of answer quality and source grounding
* Background indexing for larger document collections
* More robust document versioning and index replacement
* Deployment and operational monitoring

## Learning Outcomes

This project provided practical experience with:

* Designing a modular Python application
* Building a complete RAG pipeline
* Working with embeddings and vector databases
* Integrating language models through LangChain
* Managing configuration and secrets
* Writing unit tests with mocks
* Building an interactive Streamlit application
* Implementing basic document validation, error handling, and logging
