# RAG Evaluation with RAGAS and LangFuse

This project demonstrates the evaluation of a Retrieval-Augmented Generation (RAG) system using RAGAS (Retrieval Augmented Generation Assessment System) and LangFuse for performance tracking. It includes code for generating test datasets, evaluating RAG performance, and visualizing results. The project is available both as a Jupyter notebook and as a FastAPI application.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Jupyter Notebook](#jupyter-notebook)
   - [API](#api)
5. [Project Structure](#project-structure)
6. [Key Components](#key-components)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Visualization](#visualization)
9. [LangFuse Integration](#langfuse-integration)
10. [Contributing](#contributing)
11. [License](#license)

## Project Overview

This project aims to evaluate the performance of a RAG system using various metrics provided by RAGAS. It also integrates LangFuse for detailed performance tracking and analysis. The main steps include:

1. Loading and preprocessing documents
2. Generating a test dataset
3. Evaluating the RAG system using RAGAS metrics
4. Visualizing the results
5. Tracking performance with LangFuse

## Prerequisites

- Python 3.7+
- OpenAI API key
- LangFuse account (for performance tracking)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rag-evaluation-project.git
   cd rag-evaluation-project
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   LANGFUSE_HOST=http://localhost:3005
   API_URL=http://localhost:8000/query
   PROJECT=bhzgraph_new
   ```

## Usage

### Jupyter Notebook

1. Ensure your document content is in `bhz_content.txt`.
2. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```
3. Open and run `ragas.ipynb`.

The notebook will guide you through the process of loading data, generating a test dataset, evaluating the RAG system, and visualizing the results.

### API

1. Ensure your document content is in `bhz_content.txt`.
2. Start the FastAPI server:
   ```
   python api.py
   ```
3. The API will be available at `http://localhost:8001`.

#### API Endpoints

##### POST /evaluate

Evaluates the RAG system using RAGAS metrics.

**Parameters:**

|
 Name      
|
 Type 
|
 Description                                   
|
 Required 
|
|
-----------
|
------
|
-----------------------------------------------
|
----------
|
|
 test_size 
|
 int  
|
 Number of test questions to generate (default: 10) 
|
 No       
|

**Request Body:**

```json
{
  "test_size": 15
}
```

**Response:**

The API returns a JSON object containing:

- `result`: A dictionary of RAGAS evaluation metrics
- `heatmap`: A base64-encoded string of the heatmap visualization

**Example Requests:**

1. Basic evaluation with default test size:
   ```
   curl -X POST "http://localhost:8001/evaluate" \
        -H "Content-Type: application/json"
   ```

2. Evaluation with custom test size:
   ```
   curl -X POST "http://localhost:8001/evaluate" \
        -H "Content-Type: application/json" \
        -d '{"test_size": 15}'
   ```

**Example Response:**

```json
{
  "result": {
    "context_precision": 0.85,
    "context_recall": 0.78,
    "faithfulness": 0.92,
    "answer_relevancy": 0.89
  },
  "heatmap": "base64_encoded_image_string"
}
```

## Project Structure

- `ragas.ipynb`: Jupyter notebook for RAG evaluation
- `api.py`: FastAPI application for RAG evaluation
- `app/`: Directory containing modules for evaluation and test set generation
- `bhz_content.txt`: Source document for RAG system
- `requirements.txt`: List of Python dependencies
- `.env`: Environment variables (not included in repository)

## Key Components

1. **Document Loading and Chunking**: Uses LangChain's TextLoader and SemanticChunker.
2. **Language Detection**: Utilizes OpenAI's ChatGPT to detect the language of the content.
3. **Test Set Generation**: Employs RAGAS's TestsetGenerator to create evaluation datasets.
4. **RAG System**: Implements a simple API-based RAG system for testing.
5. **Evaluation**: Uses RAGAS metrics for comprehensive evaluation.
6. **Visualization**: Creates heatmaps of evaluation results using Seaborn.
7. **Performance Tracking**: Integrates LangFuse for detailed performance analysis.

## Evaluation Metrics

The project uses the following RAGAS metrics:
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy

## Visualization

The project generates a heatmap visualization of the evaluation metrics for each question in the test set. In the Jupyter notebook, this is displayed inline. In the API, it's returned as a base64-encoded image.

## LangFuse Integration

LangFuse is used to track and analyze the performance of the RAG system. It provides detailed insights into each evaluation metric.

## Contributing

We welcome contributions to this project. If you have any ideas, suggestions, or improvements, please create a pull request or open an issue on the repository.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any questions or inquiries, please contact us at [sebastian.schkudlara@gmail.com].
# ragas_app
# ragas_app
# ragas_app
