from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from langchain_openai import ChatOpenAI
from app.utils import clear_ragas_cache
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.pydantic_v1 import ValidationError
import logging

logger = logging.getLogger(__name__)

def generate_testset(chunks, detected_language, test_size, distributions):
    generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)
    MAX_RETRIES = 3
    for retry_count in range(MAX_RETRIES):
        try:
            generator.adapt(detected_language, evolutions=[simple, reasoning, conditional, multi_context])
            generator.save(evolutions=[simple, reasoning, multi_context, conditional])
            break
        except ValidationError as e:
            error_message = str(e)
            if "Expecting value" in error_message or "self.extractor_prompt = self.extractor_prompt.adapt(" in error_message:
                logger.info(f"Invalid JSON format or prompt adaptation error on attempt {retry_count + 1}. Clearing cache and retrying...")
                clear_ragas_cache()
            else:
                raise e
    else:
        raise RuntimeError("Exceeded maximum retries. You may need to clear the cache manually.")

    return generator.generate_with_langchain_docs(
        chunks,
        test_size=test_size,
        distributions={
            simple: distributions.get("simple", 0.4),
            reasoning: distributions.get("reasoning", 0.2),
            multi_context: distributions.get("multi_context", 0.2),
            conditional: distributions.get("conditional", 0.2)
        },
        is_async=True
    )
