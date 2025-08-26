import logging, requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)

# ✅ Use LangChain's Ollama integration
from langchain_community.llms import Ollama as LangchainOllama
from langchain_community.embeddings import OllamaEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


API_URL = "http://localhost:5601/v1/chat/completions"

def query_rag_app(question: str) -> str:
    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "user", "content": question}
        ]
    }
    response = requests.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    # Extract assistant reply
    return data["choices"][0]["message"]["content"]


def build_dataset():
    # Ground-truth Q&A pairs for evaluation
    questions = [
        "What are the disciplinary actions in the penalty clauses?",
        "Which financial rules violations lead to fines?"
    ]
    ground_truths = [
        "Penalty clauses specify suspension and fines for violations.",
        "Fines are imposed for financial misconduct."
    ]

    # Run queries against your RAG app
    answers = []
    contexts = []  # Optionally you can log retrieved chunks here

    for q in questions:
        ans = query_rag_app(q)
        answers.append(ans)
        contexts.append([])  # if your API can return retrieved docs, put them here

    data = {
        "question": questions,
        "answer": answers,        # from your system
        "contexts": contexts,     # keep empty or populate later
        "ground_truth": ground_truths
    }
    return Dataset.from_dict(data)


def run_evaluation():
    dataset = build_dataset()

    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]

    logger.info("Running RAGAs evaluation with Ollama...")

    # LangChain wrappers (compatible with Ragas)
    ollama_llm = LangchainOllama(model="llama3.2:3b", base_url="http://localhost:11434")
    ollama_embed = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

    # Pass into ragas evaluate
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=ollama_llm,
        embeddings=ollama_embed
    )

    logger.info("Evaluation complete.")

    # Per-sample results
    df = result.to_pandas()
    print("📊 Per-sample evaluation results:")
    print(df)

    # Aggregate scores
    print("\n Aggregate Scores:")
    for metric_name, score in result.scores:
        print(f"  {metric_name}: {score:.3f}")


if __name__ == "__main__":
    run_evaluation()
