# LlamaIndex RAG Application with Ollama and PGVector

This project implements a basic Retrieval-Augmented Generation (RAG) pipeline using LlamaIndex, Ollama, and PostgreSQL with the PGVector extension. It allows you to ingest documents, generate embeddings locally with Ollama, store them in a PostgreSQL database, and query your data using a powerful Large Language Model (LLM) also powered by Ollama.

## Features

*   **Document Ingestion:** Load and parse various document types (PDFs, DOCX, etc.) from a local directory.
*   **Local Embeddings:** Generate document embeddings using an Ollama-hosted embedding model (`nomic-embed-text`).
*   **Vector Storage:** Persist document embeddings in a PostgreSQL database using the PGVector extension.
*   **Local LLM Integration:** Utilize a local Large Language Model (LLM) (e.g., `llama3.2:3b`) hosted on Ollama for query responses.
*   **Interactive Querying:** A basic query engine to interact with your indexed documents.

## Technologies Used

*   **[LlamaIndex](https://www.llamaindex.ai/):** Data framework for LLM-powered applications.
*   **[Ollama](https://ollama.com/):** A framework for running open-source LLMs and embedding models locally.
*   **[PostgreSQL](https://www.postgresql.org/):** A powerful, open-source object-relational database system.
*   **[PGVector](https://github.com/pgvector/pgvector):** PostgreSQL extension for storing vector embeddings.
*   **[Python](https://www.python.org/):** The primary programming language used.

## Setup Instructions

### 1. Install Prerequisites

*   **Ollama:** Download and install Ollama from the official website. Pull the required models:
    ```bash
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
    ```
*   **PostgreSQL:** Install PostgreSQL and ensure the `pgvector` extension is enabled in your database. You might need to run `CREATE EXTENSION IF NOT EXISTS vector;` in your PostgreSQL database, {Link: according to LlamaIndex https://docs.llamaindex.ai/en/v0.10.20/examples/vector_stores/postgres.html}.
*   **Python:** Ensure you have Python installed (preferably Python 3.9+).

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
.\venv\Scripts\activate # On Windows
source venv/bin/activate # On macOS/Linux
