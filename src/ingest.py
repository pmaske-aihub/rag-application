# ingest.py
import logging
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
from psycopg2 import sql, connect

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
        # Create a directory named 'data' and place your text documents (e.g., .txt, .pdf) inside it.
        documents = SimpleDirectoryReader(input_dir="./src/data").load_data()
        logger.info(f"Loaded {len(documents)} documents.")
        # logger.debug(f"First document content: {documents[0].text[:200]}...") # Log first 200 chars
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
            # If the table doesn't exist, it will be created by LlamaIndex
        )
        logger.info("PGVectorStore instance created.")
    except Exception as e:
        logger.error(f"Error connecting to PGVectorStore: {e}", exc_info=True)
        exit(1)


    # 4. Build the LlamaIndex Index
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        # Build the index from the loaded documents and store it in PGVector.
        # This will create the embeddings and insert them into the 'documents' table.
        vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[node_parser],
        )
        logger.info("LlamaIndex built and stored in PGVector.")
        
        # If you have an existing index and are only adding new documents,
        # you would typically load the index first and then insert.
        # For a fresh build or re-ingestion, from_documents is fine.
        # The loop below is redundant if from_documents was just called on all docs.
        # If you were adding *new* documents to an *existing* index:
        # vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        # for doc in new_documents_to_add:
        #     vector_index.insert(doc)

    except Exception as e:
        logger.error(f"Error building or ingesting into LlamaIndex: {e}", exc_info=True)
        exit(1)

    # 5. Create a Basic Query Engine (Optional in ingest.py, good for testing)
    # Configure LlamaIndex to use the Ollama LLM.
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
        except Exception as e:
            print(f"Error during query: {e}")

