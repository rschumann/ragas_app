import nest_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
import pandas as pd
import requests
import os
import logging
from dotenv import load_dotenv
from langfuse import Langfuse
from shutil import rmtree
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import base64
from io import BytesIO
from fastapi.responses import JSONResponse
from langchain_core.pydantic_v1 import ValidationError
import traceback
import numpy as np
from datasets import Dataset, Features, Sequence, Value

# Apply nest_asyncio to allow re-entering the event loop
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define Models
generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Langfuse
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

# Read project name from the environment variables
PROJECT = os.getenv('PROJECT', 'testing')

# API Request Model
class EvaluationRequest(BaseModel):
    file_path: str  # Path to the file in the body
    test_size: int = 2  # Default test size
    chunk_method: int = 1  # 1 for semantic chunking, 2 for character-based chunking
    distributions: dict = {"simple": 0.4, "reasoning": 0.2, "multi_context": 0.2, "conditional": 0.2}

# API Response Model
class EvaluationResponse(BaseModel):
    results: dict
    langfuse_trace_id: Optional[str]

# Function to clear RAGAS cache
def clear_ragas_cache():
    ragas_cache_dir = os.path.expanduser("~/.cache/ragas")
    if os.path.exists(ragas_cache_dir):
        rmtree(ragas_cache_dir)
        logger.info("Cleared RAGAS cache")

# Function to detect language
def detect_language(document_content: str) -> str:
    detection_prompt = (
        "You are a language detection assistant. Analyze the following text and respond with the primary language of the text. "
        "Respond only with the language name in English, without explanations."
    )
    text_snippet = document_content[:500]
    messages = [
        {"role": "system", "content": detection_prompt},
        {"role": "user", "content": text_snippet}
    ]
    language_detection_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    response = language_detection_llm.invoke(messages)
    return response.content.strip()

# Function to split documents based on chunking method
def chunk_documents(documents, chunk_method):
    if chunk_method == 1:
        semantic_text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="interquartile")
        chunks = semantic_text_splitter.create_documents([doc.page_content for doc in documents])
    elif chunk_method == 2:
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=600,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_documents(documents)
    else:
        raise ValueError("Invalid chunk method provided. Use 1 for semantic or 2 for character-based chunking.")
    
    for document in chunks:
        document.metadata = {'source': documents[0].metadata.get('source', 'unknown')}
    return chunks

# Function to interact with RAG API
def api_rag(question: str):
    API_URL = os.getenv('API_URL', 'http://localhost:8000/query')
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    data = {"project": PROJECT, "question": question}
    
    response = requests.post(API_URL, json=data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result["final_answer"], [result["local_answer"], result["global_answer"]]
    else:
        raise HTTPException(status_code=response.status_code, detail="RAG API request failed")

# Function to generate heatmap and bar chart, and return the images as base64 strings
def generate_visualizations(df):
    heatmap_data = df[['answer_relevancy', 'context_precision', 'context_recall', 'faithfulness']]
    cmap = LinearSegmentedColormap.from_list('green_red', ['red', 'green'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=.5, cmap=cmap)
    heatmap_img = save_plt_image()

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(df['question'])), df['answer_relevancy'])
    plt.yticks(ticks=range(len(df['question'])), labels=df['question'], rotation=0)
    plt.xlabel('Answer Relevancy')
    plt.ylabel('Questions')
    bar_chart_img = save_plt_image()

    return heatmap_img, bar_chart_img

# Helper function to save the current matplotlib figure as a base64 string
def save_plt_image():
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()  # Close the figure to free up memory
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag(request: EvaluationRequest):
    try:
        # Load and process document
        loader = TextLoader(request.file_path)
        documents = loader.load()

        if not documents or len(documents) == 0:
            raise HTTPException(status_code=400, detail="No documents were loaded from the provided file path.")
        
        # Chunk documents based on the chunking method
        chunks = chunk_documents(documents, request.chunk_method)
        
        if len(chunks) == 0:
            raise HTTPException(status_code=400, detail="No chunks were created from the document.")
        
        # Detect the language of the first chunk
        detected_language = detect_language(chunks[0].page_content)
        
        # Initialize TestsetGenerator
        generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)
        
        # Retry logic with cache clearing for generator adaptation
        MAX_RETRIES = 3
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                generator.adapt(detected_language, evolutions=[simple, reasoning, conditional, multi_context])
                generator.save(evolutions=[simple, reasoning, multi_context, conditional])
                break  # If successful, exit the loop
            except ValidationError as e:  # Handle prompt adaptation or other errors
                error_message = str(e)
                if "Expecting value" in error_message or "self.extractor_prompt = self.extractor_prompt.adapt(" in error_message:
                    logger.info(f"Invalid JSON format or prompt adaptation error on attempt {retry_count + 1}. Clearing cache and retrying...")
                    clear_ragas_cache()
                    retry_count += 1
                else:
                    raise e

        if retry_count == MAX_RETRIES:
            raise RuntimeError("Exceeded maximum retries. You may need to clear the cache manually.")
        
        # Generate test set
        testset = generator.generate_with_langchain_docs(
            chunks,
            test_size=request.test_size,
            distributions={
                simple: request.distributions.get("simple", 0.4),
                reasoning: request.distributions.get("reasoning", 0.2),
                multi_context: request.distributions.get("multi_context", 0.2),
                conditional: request.distributions.get("conditional", 0.2)
            },
            is_async=True
        )
        
        # Prepare data for evaluation
        data = {"question": [], "answer": [], "contexts": [], "ground_truth": testset.to_pandas()["ground_truth"].to_list()}
        
        # Retrieve answers for each question using the RAG API
        for query in testset.to_pandas()["question"].to_list():
            answer, contexts = api_rag(query)
            data["question"].append(query)
            data["answer"].append(answer)
            data["contexts"].append(contexts)

        # Convert to pandas DataFrame and create a dataset
        df = pd.DataFrame(data)

        # Prepare dataset for evaluation
        df['retrieved_contexts'] = df['contexts']  # 'contexts' are the retrieved contexts
        df['user_input'] = df['question']  # 'question' is the user input
        df['response'] = df['answer']  # 'answer' is the response
        df['reference'] = df['ground_truth']  # 'ground_truth' is the reference

        # Convert the dataframe to Hugging Face Dataset
        dataset = Dataset.from_pandas(df)

        # Evaluate the dataset
        result = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
        )

        df = result.to_pandas()
        df['question'] = df['user_input']

        # Convert result DataFrame to a JSON-serializable format
        result_dict = df.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).to_dict(orient="records")

        # Generate the visualizations (heatmap and bar chart)
        heatmap_img, bar_chart_img = generate_visualizations(df)
        
        # Log evaluation results in LangFuse
        trace = langfuse.trace(
            name="Evaluation Trace",
            user_id="user_id_placeholder",
            metadata={"source_file": request.file_path},
            tags=["evaluation"]
        )
        
        # Log scores to LangFuse
        for _, row in result.to_pandas().iterrows():
            for metric_name in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
                langfuse.score(
                    name=metric_name,
                    value=row[metric_name],
                    trace_id=trace.id
                )

        # Prepare a list to hold only the metrics
        metrics_list = []
        for _, row in df.iterrows():  # Use 'df' instead of 'result_df'
            metrics = {
                "question": row["user_input"],
                "faithfulness": row["faithfulness"],
                "answer_relevancy": row["answer_relevancy"],
                "context_recall": row["context_recall"],
                "context_precision": row["context_precision"]
            }
            metrics_list.append(metrics)

        # Return metrics_list in the results field
        return JSONResponse(content={
            "results": metrics_list,  # List of metrics per question
            "langfuse_trace_id": trace.id,
            "heatmap_image": heatmap_img,  # Base64 encoded heatmap image
            "bar_chart_image": bar_chart_img  # Base64 encoded bar chart image
        })

    except Exception as e:
        # Log the stack trace for easier debugging
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import sys
    if "uvicorn" in sys.modules:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    else:
        import logging
        logging.basicConfig(level=logging.INFO)
        app.run(host="0.0.0.0", port=8001)
