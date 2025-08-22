# Install required libraries
# Ensure you have ollama installed and running with llama3.1:8b and nomic-embed-text models pulled
# Ensure you have PostgreSQL installed and pgvector extension enabled.

# !pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-vector-stores-postgres psycopg2-binary sqlalchemy

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.readers.file.docs import DocxReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

# 1. Load and parse documents
# Create a directory named 'data' and place your text documents (e.g., .txt, .pdf) inside it.
# You can also customize the SimpleDirectoryReader to load from other sources like URLs.
documents = SimpleDirectoryReader(input_dir="./src/data").load_data()

print(f"Loaded {len(documents)} documents.")
print(f"First document content:", documents[0]) 

# 2. Generate Embeddings with Ollama
# Configure LlamaIndex to use the Ollama embedding model.
# Make sure 'nomic-embed-text' or your chosen embedding model is pulled in Ollama.
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")

# 3. Store in PGVector
# Configure your PostgreSQL connection.
# Replace with your actual database name, host, password, port, and user.
db_name = "ragdb" # 
connection_string = f"postgresql://postgres:postgres@localhost:5432/{db_name}"
url = make_url(connection_string)


# Create the PGVectorStore
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=str(url.port),
    user=url.username,
    table_name="documents", 
    embed_dim=768, 
)

# 4. Build the LlamaIndex Index
# Configure node parser for chunking documents.
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

# Create the storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build the index from the loaded documents and store it in PGVector.
vector_index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    transformations=[node_parser],
)

# Ingest new documents into the existing index
for doc in documents: # Assuming 'documents' now contains all documents
    vector_index.insert(doc)


# 5. Create a Basic Query Engine
# Configure LlamaIndex to use the Ollama LLM.
Settings.llm = Ollama(model="llama3.2:3b", request_timeout=120.0, base_url="http://localhost:11434")

# Create the query engine
query_engine = vector_index.as_query_engine()

# Ask a question
while True:
    query_text = input("Enter your query (type 'exit' to quit): ")
    if query_text.lower() == "exit":
        break
    response = query_engine.query(query_text)
    print(response)
