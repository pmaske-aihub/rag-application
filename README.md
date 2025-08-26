# Contextual RAG Chatbot with LlamaIndex, Ollama & PGVector

This project is an **interactive chatbot** built as part of an interview exercise, showcasing a modern **Contextual RAG (Retrieval-Augmented Generation) pipeline**.  
It demonstrates document ingestion, vector storage, local LLM inference, dynamic routing, evaluation, and monitoring — all running **locally** using **open-source tools**.

---

## Features

### Document Ingestion & Storage
- Supports **PDFs, DOCX**, and other document formats.
- Stored in **PostgreSQL (`ragdb`)** with **PGVector** extension for embeddings.

### Contextual RAG Pipeline
- **LlamaIndex** used to build retrievers and query engines.
- **Ollama models** (`nomic-embed-text` + `llama3.2:3b`) for embeddings and generation.
- **Dynamic Router**: Chooses between RAG and direct LLM responses (for chit-chat / non-document queries).

### Interactive Query API
- **FastAPI backend** with `/chat/completions` endpoint.
- Compatible with **Open WebUI** for interactive chatbot usage.

### Evaluation & Monitoring
- **RAGAs**: Evaluates precision, recall, faithfulness, and answer relevance.
- **Arize Phoenix**: Observability for prompts, RAG pipeline monitoring, and agent tracing.

### Extensible Design
- Ready for **Crew.AI-based prompt optimization**, rerankers, and agent orchestration.

---

## Tech Stack
- **LLM & Embeddings:** Ollama (`nomic-embed-text`, `llama3.2:3b`)
- **RAG Framework:** LlamaIndex
- **Database:** PostgreSQL with PGVector
- **Backend:** FastAPI
- **Evaluation:** RAGAs
- **Tracing & Monitoring:** Arize Phoenix
- **Chatbot interface:** Open WebUI

---

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/pmaske-aihub/rag-application.git
   cd rag-application
   ```
2. **Install Prerequisites**
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
   Ensure **Postgres** is installed, and enable `pgvector`.
   ```sql
   CREATE DATABASE ragdb;
   \c ragdb
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
   Ensure that **Open WebUI** is installed and running either locally or via Docker Desktop. See [How to install Open WebUI](https://github.com/open-webui/open-webui?tab=readme-ov-file#how-to-install-)
   
4. **Setup Environment**
   ```bash
    python -m venv venv
    .\venv\Scripts\activate   # Windows
    source venv/bin/activate  # Linux/Mac
    pip install -r requirements.txt
   ```
5. **Run the FastAPI backend**
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 5601 --workers 4
   ```
## **Access the chatbot**
   
   On the web broweser, access `http://localhost:3000`. This will open an **Open WebUI** interface. Go to Admin Panel > Settings > Connections and add locally running FastAPI app.
   
   <img width="1230" height="449" alt="{AB9E68AE-78E9-4CDA-9BEA-12DA4671E1F7}" src="https://github.com/user-attachments/assets/982493af-d02c-42a6-bffa-8a0391cc89b4" />

   Create a new **Workflow** and select model as `llama3.2.3b`
   
   <img width="860" height="259" alt="{2E86E0A2-DCBD-4F54-8DAB-43EA0A3EBC96}" src="https://github.com/user-attachments/assets/46d8d660-0495-4b37-8261-7cf38ef73191" />

   Select **New Chat** and switch to **Custom RAG Pipeline**
   
   <img width="794" height="404" alt="{6DD11EC6-95D1-44CB-BA2F-463780CFA9C7}" src="https://github.com/user-attachments/assets/c2076041-98cd-445b-b033-0f46a08c2413" />

   For quick test, use SwaggerUI which can be accessed on [http://localhost:5601/docs](http://localhost:5601/docs)

   <img width="1450" height="822" alt="{1DFBE23C-927C-4ED1-97FF-A801B527EC0D}" src="https://github.com/user-attachments/assets/56e7dc4d-f019-46df-9d4c-93f73e542238" />

   **Example Usage**

   `POST /chat/completions`
   
   ```json
    {
      "model": "llama3.2:3b",
      "messages": [
        {
          "role": "user",
          "content": "Based on the penalties section, what are the different levels of disciplinary actions?"
        }
      ]
    }
   ```

## **Evaluation (RAGAs)**
   ```bash
   python src/evaluate_rag.py
   ```
Outputs per-sample metrics: Context Precision / Recall, Faithfulness, Answer Relevancy.

## **Monitoring (Phoenix)**
   On the web browser, visit `http://localhost:6006`. This will open Phoenix dashboard and will show **Query pipeline traces**, **Latency breakdown** and **Prompt optimizations**.

## **Future Considerations**
- Crew.AI: Multi-agent prompt optimization.
- Rerankers (e.g., Cohere / bge-reranker).
- Docker deployment for portability.

## Acknowledgements
This project was built as part of an **interview technical exercise**, showcasing an **end-to-end RAG application** with monitoring, evaluation, and extensibility in mind.

