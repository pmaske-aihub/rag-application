# ingest.py
import logging
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
from psycopg2 import sql, connect
import os

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration (match with api.py) ---
db_name = "ragdb"
connection_string = f"postgresql://postgres:postgres@localhost:5432/{db_name}"
url = make_url(connection_string)
table_name = "documents" 
embed_dim = 768

# Ensure the pgvector extension is enabled
def ensure_pgvector_extension():
    try:
        with connect(connection_string) as conn:
            conn.autocommit = True  # Ensure autocommit is enabled for DDL
            with conn.cursor() as cur:
                cur.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS vector"))
                logger.info("Ensured pgvector extension is enabled.")
    except Exception as e:
        logger.error(f"Error ensuring pgvector extension: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    ensure_pgvector_extension()

    logger.info("Starting data ingestion process...")

    # 1. Load and parse documents
    try:
        documents = SimpleDirectoryReader(input_dir="./src/data").load_data()
        logger.info(f"Loaded {len(documents)} documents.")

        # Add file name metadata
        for doc in documents:
            if "file_name" in doc.metadata:
                doc.metadata["source"] = doc.metadata["file_name"]
            else:
                # fallback to "unknown" if file_name not captured
                doc.metadata["source"] = "unknown"

        logger.info("Attached source filename metadata to documents.")

    except Exception as e:
        logger.error(f"Error loading documents: {e}", exc_info=True)
        exit(1)

    # 2. Generate Embeddings with Ollama
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
    logger.info("Ollama Embedding configured.")

    # 3. Store in PGVector
    try:
        vector_store = PGVectorStore.from_params(
            database=db_name,
            host=url.host,
            password=url.password,
            port=str(url.port),
            user=url.username,
            table_name=table_name, 
            embed_dim=embed_dim, 
        )
        logger.info("PGVectorStore instance created.")
    except Exception as e:
        logger.error(f"Error connecting to PGVectorStore: {e}", exc_info=True)
        exit(1)

    # 4. Build the LlamaIndex Index
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[node_parser],
        )
        logger.info("LlamaIndex built and stored in PGVector with source metadata.")
    except Exception as e:
        logger.error(f"Error building or ingesting into LlamaIndex: {e}", exc_info=True)
        exit(1)

    # 5. Create a Basic Query Engine (Optional in ingest.py, good for testing)
    Settings.llm = Ollama(model="llama3.2:3b", request_timeout=120.0, base_url="http://localhost:11434")
    logger.info("Ollama LLM configured for testing.")

    query_engine = vector_index.as_query_engine()

    logger.info("Ingestion complete. You can now run api.py to serve queries.")
    logger.info("Testing query engine locally (type 'exit' to quit)...")
    while True:
        query_text = input("Enter your query (type 'exit' to quit): ")
        if query_text.lower() == "exit":
            break
        try:
            response = query_engine.query(query_text)
            print(f"Response: {response}")

            # Print sources for debugging
            if hasattr(response, "source_nodes"):
                sources = [node.metadata.get("source", "unknown") for node in response.source_nodes]
                print(f"Sources: {', '.join(set(sources))}")

        except Exception as e:
            print(f"Error during query: {e}")
