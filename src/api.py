# api.py (Updated with Sources in Response)
from fastapi import FastAPI, HTTPException
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

app = FastAPI(
    title="LlamaIndex RAG API",
    description="API to query LlamaIndex RAG built with Ollama and PGVector"
)

# --- Pydantic Models (OpenAI Compatibility) ---
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)

class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: Message

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-custom"
    object: str = "chat.completion"
    created: int = 0
    model: str
    choices: List[ChatCompletionResponseChoice]

# --- Globals ---
query_engine = None

# --- Configuration ---
db_name = "ragdb"
connection_string = f"postgresql://postgres:postgres@localhost:5432/{db_name}"
url = make_url(connection_string)
table_name = "documents"
embed_dim = 768

@app.on_event("startup")
async def startup_event():
    """Initialize the LlamaIndex query engine on FastAPI startup."""
    global query_engine
    logger.info("Initializing LlamaIndex RAG system...")

    try:
        # Embedding + LLM
        Settings.embed_model = OllamaEmbedding(
            model_name="nomic-embed-text", base_url="http://localhost:11434"
        )
        Settings.llm = Ollama(
            model="llama3.2:3b", request_timeout=120.0, base_url="http://localhost:11434"
        )

        # PGVector
        vector_store = PGVectorStore.from_params(
            database=db_name,
            host=url.host,
            password=url.password,
            port=str(url.port),
            user=url.username,
            table_name=table_name,
            embed_dim=embed_dim,
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context
        )

        query_engine = vector_index.as_query_engine()
        logger.info("Query engine created successfully.")

    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize LlamaIndex RAG system: {e}"
        )

@app.get("/")
async def read_root():
    return {"message": "LlamaIndex RAG API is running. Use /chat/completions to ask questions."}

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": "llama3.2:3b"},
            {"id": "nomic-embed-text"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if query_engine is None:
        raise HTTPException(status_code=503, detail="RAG system not ready.")

    # Extract latest user query
    user_query = next((m.content for m in request.messages if m.role == "user"), None)
    if not user_query:
        raise HTTPException(status_code=400, detail="No user query found in messages.")

    logger.info(f"User query: {user_query}")

    try:
        response = query_engine.query(user_query)
        rag_response_content = str(response)

        # Collect sources
        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                src = node.metadata.get("source", None)
                if src:
                    sources.append(src)

        sources = list(set(sources))
        if sources:
            rag_response_content += f"\n\nSources: {', '.join(sources)}"

        print('Sources:', sources[:5])  # Log first 5 sources for debugging

        # Return OpenAI-compatible response
        return ChatCompletionResponse(
            model=request.model,
            choices=[ChatCompletionResponseChoice(
                message=Message(role="assistant", content=rag_response_content)
            )]
        )

    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {e}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5601)
