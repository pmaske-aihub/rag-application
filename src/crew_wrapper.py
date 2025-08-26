from crewai import LLM, Agent, Task, Crew
from llama_index.llms.ollama import Ollama
import logging
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from phoenix.otel import register


# setup monitoring for your crew
tracer_provider = register(endpoint="http://localhost:6006/v1/traces")
CrewAIInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptOptimizer:
    def __init__(self, query_engine):
        self.query_engine = query_engine

    def optimize(self, query: str) -> str:
        """
        Optimize a user query into a single clean query string.
        Ensures no JSON or multi-form outputs are returned.
        """
        try:
            # Example using CrewAI to refine the prompt
            optimized = query.strip()

            # Example (if using CrewAI): ensure it always picks the first form
            # forms = crewai_generate_forms(query)
            # optimized = forms[0] if forms else query.strip()

            logger.info(f"Optimized query: {optimized}")
            return optimized
        except Exception as e:
            logger.warning(f"Prompt optimization failed, falling back to raw query: {e}")
            return query.strip()