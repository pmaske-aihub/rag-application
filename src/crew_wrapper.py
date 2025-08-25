from crewai import LLM, Agent, Task, Crew
from llama_index.llms.ollama import Ollama
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptOptimizer:
    def __init__(self, query_engine):
        self.query_engine = query_engine

    def optimize_and_query(self, user_query: str):
        optimizer = Agent(
            role="Prompt Optimizer",
            goal="Rewrite and expand user queries to improve retrieval quality",
            backstory="Helps improve recall in RAG pipelines",
            llm="ollama/llama3.2:3b"
        )

        task = Task(
            description=f"Rewrite the query '{user_query}' into 2 better forms for improved retrieval.",
            agent=optimizer,
            expected_output="2 optimized query forms"
        )

        crew = Crew(agents=[optimizer], tasks=[task])
        optimized = crew.kickoff()

        logger.info(f"PromptOptimizer produced: {optimized}")

        # For simplicity: just run original + optimized query combined
        final_query = f"{user_query}\n\nOptimized: {optimized}"
        response = self.query_engine.query(final_query)
        return response