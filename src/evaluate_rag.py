import logging
import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall
    # faithfulness, # excluding for demo purpose as it is breaking complex JSON resulting in timeout error.
    # answer_relevancy # excluding for demo purpose as it is breaking complex JSON resulting in timeout error.
)

# Use LangChain's new Ollama integration
from langchain_ollama import OllamaLLM, OllamaEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:5601/v1/chat/completions"

def query_rag_app(question: str):
    """Send a query to the RAG API and return assistant response + contexts."""
    payload = {
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": question}]
    }
    response = requests.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    answer = data["choices"][0]["message"]["content"]

    # Try to extract contexts (sources) if returned
    sources = []
    if "Sources:" in answer:
        try:
            sources = answer.split("Sources:")[-1].strip().split(",")
            sources = [s.strip() for s in sources]
        except Exception:
            sources = []

    return answer, sources


def build_dataset():
    questions = [
        "What are the disciplinary actions in the penalty clauses?",
        "Which financial rules violations lead to fines?"
    ]
    ground_truths = [
        "Penalty clauses specify suspension and fines for violations.",
        "Fines are imposed for financial misconduct."
    ]

    answers, contexts = [], []

    for q in questions:
        ans, ctx = query_rag_app(q)
        answers.append(ans)
        contexts.append(ctx)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })


def run_evaluation():
    dataset = build_dataset()

    #metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    metrics = [context_precision, context_recall]

    logger.info("Running RAGAs evaluation with Ollama...")

    # Correct LangChain wrappers (works with RAGAS) since RAGAS does not work with llama_index. Using model with 3b for better performance.
    ollama_llm = OllamaLLM(model="llama3.2:3b", base_url="http://localhost:11434")
    ollama_embed = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=ollama_llm,
        embeddings=ollama_embed,
    )

    logger.info("Evaluation complete.")

    # Per-sample results
    df = result.to_pandas()
    print("Per-sample evaluation results:")
    print(df)

if __name__ == "__main__":
    run_evaluation()
