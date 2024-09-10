# %% [markdown]
# # Block 1, Load Data from file

# %%
import requests
import os
import logging
import shutil
from dotenv import load_dotenv
import json
import time
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.pydantic_v1 import ValidationError
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset, Features, Sequence, Value
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from langfuse import Langfuse

# %% [markdown]
# Load logger and environment variables

# %%
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %% [markdown]
# Define clear_ragas_cache

# %%
def clear_ragas_cache():
    ragas_cache_dir = os.path.expanduser("~/.cache/ragas")
    if os.path.exists(ragas_cache_dir):
        shutil.rmtree(ragas_cache_dir)
        logger.info("Cleared RAGAS cache")

# %% [markdown]
# Define Models

# %%
# Define your chat models
generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
# Define your embedding models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# %% [markdown]
# Load the document

# %%
file_path = "../files/testing.txt"
file_name = "testing.txt"
loader = TextLoader(file_path)
documents = loader.load()

# %% [markdown]
# Initialize the SemanticChunker and TextSplitter with OpenAI embeddings

# %%
semantic_text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="interquartile")
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=600,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# %%
print(semantic_text_splitter)
print(text_splitter)

# %% [markdown]
# Split documents method using semantic meaning or simple splitter

# %%
def chunk_documents(documents, chunk_method):
    if chunk_method == 1:
        # Use the semantic text splitter
        chunks = semantic_text_splitter.create_documents([doc.page_content for doc in documents])
    elif chunk_method == 2:
        # Use the regular text splitter
        chunks = text_splitter.split_documents(documents)
    else:
        raise ValueError("Invalid chunk method provided. Use 1 for semantic text splitter, 2 for regular text splitter.")
    return chunks

# %% [markdown]
# Split documents using text splitter

# %%
chunks = chunk_documents(documents, 1)

# %% [markdown]
# Manually add 'file_name' metadata to the chunks

# %%
for document in chunks:
    # Ensure that the document has a metadata field
    if not hasattr(document, 'metadata') or document.metadata is None:
        document.metadata = {}

    # Set or update the 'source' field
    document.metadata['source'] = file_name

# %% [markdown]
# # Block 2, RAGAS expects a file_name dict as key

# %%
for document in chunks:
    document.metadata['file_name'] = document.metadata['source']
    document.metadata['filename'] = document.metadata['source']

# %% [markdown]
# # Block 3, Detect language using OpenAI's ChatOpenAI

# %% [markdown]
# Define the function to detect the main language using OpenAI

# %%
def detect_language(document_content: str) -> str:
    # Create a prompt for detecting the language
    detection_prompt = (
        "You are a language detection assistant. Analyze the following text and respond with the primary language of the text. "
        "Respond only with the language name in English, without explanations."
    )

    # Limit the document content to the first 500 characters
    text_snippet = document_content[:500]

    # Create the messages that will be passed to the model
    messages = [
        {"role": "system", "content": detection_prompt},
        {"role": "user", "content": text_snippet}
    ]

    # Invoke OpenAI model to detect the language
    language_detection_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    response = language_detection_llm.invoke(messages)

    # Extract the detected language directly from the response content
    detected_language = response.content.strip()
    logger.info(f"Detected language: {detected_language}")
    return detected_language

# %% [markdown]
# Detect Language

# %%
first_chunk_text = chunks[0].page_content  # Assuming you have `chunks` from your previous code
detected_language = detect_language(first_chunk_text)

# %% [markdown]
# # Block 4, Generate Testset using ragas

# %% [markdown]
# Initialize the test set generator

# %%
generator = TestsetGenerator.from_langchain(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings
)

# %% [markdown]
# Adapt to the detected language and save the prompt adaptations for future use

# %%
print(chunks)

# %%
MAX_RETRIES = 3
retry_count = 0

while retry_count < MAX_RETRIES:
    try:
        # Attempt to adapt and save generator
        generator.adapt(detected_language, evolutions=[simple, reasoning, conditional, multi_context])
        generator.save(evolutions=[simple, reasoning, multi_context, conditional])
        break  # If successful, exit the loop
    except ValidationError as e: # type: ignore
        # Check if error is related to invalid JSON or prompt adaptation
        error_message = str(e)
        if "Expecting value" in error_message or "self.extractor_prompt = self.extractor_prompt.adapt(" in error_message:
            logger.info(f"Invalid JSON format or prompt adaptation error on attempt {retry_count + 1}. Clearing cache and retrying...")
            clear_ragas_cache()  # Clear the RAGAS cache
            retry_count += 1
        else:
            # If it's another kind of validation error, re-raise it
            raise e

if retry_count == MAX_RETRIES:
    raise RuntimeError("Exceeded maximum retries. You may need to clear the cache manually.")

# %%
distributions = {
    simple:0.4,
    reasoning:0.2,
    multi_context:0.2,
    conditional:0.2,
}
    # Sleep for 5 seconds before attempting
time.sleep(2)

# Generate testset with langchain docs
testset = generator.generate_with_langchain_docs(
    chunks,
    test_size=2,
    distributions=distributions,
    with_debugging_logs=True,
    is_async=True,
)

# %%
print(chunks)

# %% [markdown]
# # Block 5, Convert testset to JSON format and print

# %%
test_data = testset.test_data

# %% [markdown]
# Convert each DataRow to a dictionary

# %%
test_data_dicts = []
for data_row in test_data:
    test_data_dicts.append({
        "question": data_row.question,
        "contexts": data_row.contexts,
        "ground_truth": data_row.ground_truth,
        "evolution_type": data_row.evolution_type,
        "metadata": data_row.metadata
    })

# %% [markdown]
# Convert the list of dictionaries to JSON

# %%
json_testset = json.dumps(test_data_dicts, indent=2, default=str)

# %% [markdown]
# Print the JSON

# %%
print(json_testset)

# %% [markdown]
# # Block 6, Convert testset to pandas and print

# %%
testset_pandas = testset.to_pandas()
print(testset_pandas)

# %% [markdown]
# # Block 7: Set up API-based retriever and answerer

# %% [markdown]
# Get API URL and project name from environment variables

# %%
API_URL = os.getenv('API_URL', 'http://localhost:8000/query')
PROJECT = os.getenv('PROJECT', 'testing')

# %%
def api_rag(question):
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
    }
    data = {
        "project": PROJECT,
        "question": question
    }
    response = requests.post(API_URL, json=data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result["final_answer"], [result["local_answer"], result["global_answer"]]
    else:
        print(f"Error: API request failed with status code {response.status_code}")
        return "", ["", ""]

# %% [markdown]
# # Block 8, Create RAG chain

# %%
questions = testset.to_pandas()["question"].to_list()
ground_truth = testset.to_pandas()["ground_truth"].to_list()

# %% [markdown]
# # Block 9: Create dataset for evaluation

# %%
data = {"question": [], "answer": [], "contexts": [], "ground_truth": ground_truth}

# %%
for query in questions:
    answer, contexts = api_rag(query)
    data["question"].append(query)
    data["answer"].append(answer)
    data["contexts"].append(contexts)  # This is already a list of two strings

# %% [markdown]
# Convert to pandas DataFrame first

# %%
df = pd.DataFrame(data)

# %% [markdown]
# Create the dataset from the DataFrame

# %%
dataset = Dataset.from_pandas(df)

# %% [markdown]
# # Block 10: Prepare dataset and Evaluate RAG chain

# %% [markdown]
# Prepare the dataset

# %%
# Step 1: Convert dataset to dictionary
data_dict = dataset.to_dict()

# Step 2: Ensure 'contexts' and 'retrieved_contexts' are lists of lists
data_dict['contexts'] = [[context] if isinstance(context, str) else context for context in data_dict['contexts']]
data_dict['retrieved_contexts'] = [[context] if isinstance(context, str) else context for context in data_dict['contexts']]

# Step 3: Populate required columns
data_dict['user_input'] = data_dict['question']
data_dict['response'] = data_dict['answer']
data_dict['reference'] = data_dict['ground_truth']

# %% [markdown]
# Create a new dataset with the correct feature specification

# %%
prepared_dataset = Dataset.from_dict(data_dict)

# %% [markdown]
# Verify the dataset structure

# %%
print(prepared_dataset)
print(prepared_dataset.features)

# %% [markdown]
# Evaluate using the prepared dataset

# %%
result = evaluate(
    dataset = prepared_dataset,
    metrics=[
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    ],
)

# %% [markdown]
# # Block 11, result to pandas

# %%
df = result.to_pandas()
df['question'] = df['user_input']

# %% [markdown]
# # Block 12, Plot heatmap

# %%
heatmap_data = df[['answer_relevancy', 'context_precision', 'context_recall', 'faithfulness']]

# %%
cmap = LinearSegmentedColormap.from_list('green_red', ['red', 'green'])

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=.5, cmap=cmap)

# %%
# Plot some example data (e.g., answer_relevancy)
plt.barh(range(len(df['question'])), df['answer_relevancy'])

# Set the yticks with questions
plt.yticks(ticks=range(len(df['question'])), labels=df['question'], rotation=0)

plt.xlabel('Answer Relevancy')
plt.ylabel('Questions')
plt.tight_layout()
plt.show()

# %% [markdown]
# # Use LangFuse for Stats

# %% [markdown]
# # Block 13: Add LangFuse

# %% [markdown]
# Initialize Langfuse with environment variables

# %%
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

# %% [markdown]
# Verify the configuration

# %%
print("Langfuse configuration:")
print(f"Host: {os.getenv('LANGFUSE_HOST')}")
print(f"Public Key: {os.getenv('LANGFUSE_PUBLIC_KEY')}")
print("Secret Key: ****" + os.getenv("LANGFUSE_SECRET_KEY")[-4:])  # Print only last 4 characters for security

# %% [markdown]
# # Block 14, Add LangFuse

# %%
trace = langfuse.trace(
    name = "Sebastian Schkudlara",
    user_id = "cm0p49bzf0000k7wx3qmhspnr",
    metadata = {
        "email": "sebastian@makakoo.com",
    },
    tags = ["evaluation"]
)

# %% [markdown]
# # Block 15, Add LangFuse

# %%
for _, row in df.iterrows():
    for metric_name in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        langfuse.score(
            name=metric_name,
            value=row[metric_name],
            trace_id=trace.id
        )


