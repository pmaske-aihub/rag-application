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
from .router import RAGRouter
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

# -------------------------------
# Phoenix Instrumentation
# -------------------------------
tracer_provider = register()
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(
    title="LlamaIndex RAG API",
    description="Contextual RAG API with Ollama + PGVector"
)

# -------------------------------
# Pydantic Models
# -------------------------------
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

# -------------------------------
# Globals & Config
# -------------------------------
query_engine = None
router = None

db_name = "ragdb"
connection_string = f"postgresql://postgres:postgres@localhost:5432/{db_name}"
url = make_url(connection_string)
table_name = "documents"
embed_dim = 768

# -------------------------------
# Startup Init
# -------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize LlamaIndex + PGVector on startup"""
    global query_engine, router
    logger.info("Initializing RAG system...")

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
        retriever = vector_index.as_retriever()

        # Router for contextual decision
        router = RAGRouter(retriever, query_engine, Settings.llm, threshold=0.5)

        logger.info("RAG system initialized successfully.")

    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to initialize RAG system")

# -------------------------------
# Endpoints
# -------------------------------
@app.get("/")
async def root():
    return {"message": "RAG API is running. Use /v1/chat/completions."}

@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "llama3.2:3b"}, {"id": "nomic-embed-text"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if query_engine is None or router is None:
        raise HTTPException(status_code=503, detail="RAG system not ready.")

    # Extract latest user query
    user_query = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if not user_query:
        raise HTTPException(status_code=400, detail="No user query found in messages.")

    logger.info(f"User query: {user_query}")

    try:
        # --- Simple chit-chat handling (skip RAG for greetings/small-talk) ---
        if user_query.lower() in ["hi", "hello", "hey", "hello there", "how are you?"]:
            reply = "Hello 👋! I'm your AI assistant. You can ask me questions about the documents I've been trained on."
            return ChatCompletionResponse(
                model=request.model,
                choices=[ChatCompletionResponseChoice(
                    message=Message(role="assistant", content=reply)
                )]
            )

        # --- Route & answer ---
        result = router.decide_and_answer(user_query)

        # Build response with sources (file name + page)
        rag_response_content = result["answer"]
        if result.get("sources"):
            formatted_sources = [f"{s['file_name']} (page {s['page_label']})"
                                 for s in result["sources"] if "file_name" in s]
            if formatted_sources:
                rag_response_content += f"\n\nSources: {', '.join(formatted_sources)}"

        return ChatCompletionResponse(
            model=request.model,
            choices=[ChatCompletionResponseChoice(
                message=Message(role="assistant", content=rag_response_content)
            )]
        )

    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

# -------------------------------
# Main Entrypoint
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5601, workers=4)
