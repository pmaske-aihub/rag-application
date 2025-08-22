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
*   **PostgreSQL:** Install PostgreSQL and ensure the `pgvector` extension is enabled in your database. You might need to run `CREATE EXTENSION IF NOT EXISTS vector;` in your PostgreSQL database, {https://github.com/pgvector/pgvector?tab=readme-ov-file#getting-started}.
*   **Python:** Ensure you have Python installed (preferably Python 3.9+).

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
.\venv\Scripts\activate # On Windows
source venv/bin/activate # On macOS/Linux
```

### 3. Install Python Dependencies
```bash
pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-vector-stores-postgres psycopg2-binary sqlalchemy docx2txt
```
### 4. Database Configuration

* Ensure your PostgreSQL database (`ragdb` in the example) is created and accessible.
* Verify that your database user (`postgres` in the example) has the necessary permissions to create tables and write data.

## Usage
* Clone the repo or extract the zip file.
* Place the pdf and docx files into `./src/data` directory.
* After activating the virtual environment on VS Code select Run > Run without Debugging (or Ctrl+F5)
* Open new terminal and run the fastapi app
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 5601
```
* Visit `http://localhost:5601/docs` on the web browser to access the Swagger UI.
* In the POST /chat/completions endpoint, execute the sameple request body
```bash
{
  "model": "llama3.2:3b",
  "messages": [
    {
      "role": "user",
      "content": "Based on the penalties section, what are the different levels of disciplinary actions for various infringements, and how do they relate to preserving pension or bonus rights?"
    }
  ],
  "temperature": 0.7
}
```
* Check for the request body for retrieved response. Detailed response is shown inside "content" tag.

