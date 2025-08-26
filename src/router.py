import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RAGRouter:
    def __init__(self, retriever, query_engine, llm, threshold: float = 0.5):
        self.retriever = retriever
        self.query_engine = query_engine
        self.llm = llm
        self.threshold = threshold

        # Common chit-chat patterns (expand as needed)
        self.chitchat_patterns = [
            r"^\s*hi\s*$",
            r"^\s*hello\s*$",
            r"^\s*hey\s*$",
            r"how are you",
            r"what'?s up",
            r"good (morning|afternoon|evening)",
        ]

    def _is_chitchat(self, query: str) -> bool:
        """Detect if query is simple chit-chat, not needing RAG."""
        return any(re.search(p, query.lower()) for p in self.chitchat_patterns)

    def decide_and_answer(self, query: str) -> Dict[str, Any]:
        logger.info(f"Routing decision for query: {query}")

        # Step 0: Handle chit-chat dynamically
        if self._is_chitchat(query):
            logger.info("Detected chit-chat → answering directly with LLM (no RAG).")
            answer = self.llm.complete(
                f"You are a friendly assistant. Reply casually to this greeting: {query}"
            ).text
            return {
                "answer": answer,
                "sources": [],
                "source_nodes": []
            }

        # Step 1: Retrieve
        retrieved_nodes = self.retriever.retrieve(query)
        if not retrieved_nodes:
            logger.info("No documents retrieved → falling back to direct LLM.")
            answer = self.llm.complete(query).text
            return {"answer": answer, "sources": [], "source_nodes": []}

        # Step 2: Check relevance score
        top_score = retrieved_nodes[0].score
        logger.info(f"Top retrieved doc score: {top_score}")

        if top_score < self.threshold:
            logger.info("Low relevance → using direct LLM.")
            answer = self.llm.complete(query).text
            return {"answer": answer, "sources": [], "source_nodes": []}

        # Step 3: Use RAG
        logger.info("High relevance → using RAG")
        response = self.query_engine.query(query)

        # Step 4: Collect structured sources
        sources = []
        source_nodes = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                file_name = node.metadata.get("file_name", "Unknown file")
                page_label = node.metadata.get("page_label", "N/A")
                sources.append(f"{file_name} (page {page_label})")
                source_nodes.append(node)

        return {
            "answer": str(response),
            "sources": list(set(sources)),  # deduplicate
            "source_nodes": source_nodes
        }
