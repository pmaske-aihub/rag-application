# api.py (Updated)
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LlamaIndex RAG API", description="API to query LlamaIndex RAG built with Ollama and PGVector")

# Pydantic models for OpenAI Chat Completion compatibility
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str # Open WebUI will send a model name, you can ignore or validate it
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    # Add other parameters you might want to support, like stream=True

class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: Message

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-custom"
    object: str = "chat.completion"
    created: int = 0 # Unix timestamp
    model: str
    choices: List[ChatCompletionResponseChoice]


# Global variable to hold the query engine
query_engine = None

# --- Configuration (match with ingest.py) ---
db_name = "ragdb"
connection_string = f"postgresql://postgres:postgres@localhost:5432/{db_name}"
url = make_url(connection_string)
table_name = "documents" 
embed_dim = 768 # Must match the embed_dim used during ingestion

@app.on_event("startup")
async def startup_event():
    """
    Initializes the LlamaIndex query engine on FastAPI application startup.
    This ensures the index is loaded only once when the server starts.
    """
    global query_engine
    logger.info("FastAPI startup: Initializing LlamaIndex components and loading index from PostgreSQL...")

    try:
        # 1. Configure Ollama Embedding (matching ingest.py)
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
        logger.info("Ollama Embedding configured.")

        # 2. Configure Ollama LLM (matching ingest.py)
        # Ensure this matches the model you intend to use and have pulled in Ollama.
        Settings.llm = Ollama(model="llama3.2:3b", request_timeout=120.0, base_url="http://localhost:11434")
        logger.info("Ollama LLM configured.")

        # 3. Create the PGVectorStore instance
        vector_store = PGVectorStore.from_params(
            database=db_name,
            host=url.host,
            password=url.password,
            port=str(url.port),
            user=url.username,
            table_name=table_name, 
            embed_dim=embed_dim, 
        )
        logger.info("PGVectorStore instance created (for loading).")

        # 4. Load the existing index from the vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        logger.info("LlamaIndex loaded from PGVectorStore.")

        # 5. Create the query engine
        query_engine = vector_index.as_query_engine()
        logger.info("Query engine created successfully.")

    except Exception as e:
        logger.error(f"Error during LlamaIndex initialization on startup: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize LlamaIndex RAG system: {e}. Please check the server logs."
        )

@app.get("/")
async def read_root():
    return {"message": "LlamaIndex RAG API is running. Use /chat/completions to ask questions."}

# Changed endpoint to be OpenAI compatible
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if query_engine is None:
        logger.error("Query engine not initialized. Cannot process query.")
        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready. Please check server logs for initialization errors."
        )

    # Extract the user's latest query
    user_query = ""
    for message in request.messages:
        if message.role == "user":
            user_query = message.content
    
    if not user_query:
        raise HTTPException(status_code=400, detail="No user query found in messages.")

    logger.info(f"Received OpenAI-compatible chat completion request with user query: {user_query}")

    try:
        response = query_engine.query(user_query)
        rag_response_content = str(response)

        # Format the response into an OpenAI-compatible ChatCompletionResponse
        return ChatCompletionResponse(
            model=request.model, # Use the model name sent by Open WebUI
            choices=[ChatCompletionResponseChoice(
                message=Message(role="assistant", content=rag_response_content)
            )]
        )
    except Exception as e:
        logger.error(f"Error processing query '{user_query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your query: {e}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5601)

