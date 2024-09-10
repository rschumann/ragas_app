# RAG Evaluation with RAGAS and LangFuse

This project demonstrates the evaluation of a **Retrieval-Augmented Generation (RAG)** system using **RAGAS** (Retrieval Augmented Generation Assessment System) for evaluation and **LangFuse** for performance tracking. The project provides both a Jupyter notebook and a FastAPI-based API for evaluating the performance of a RAG system. 

The evaluation metrics include:
- **Context Precision**
- **Context Recall**
- **Faithfulness**
- **Answer Relevancy**

It also generates visualizations (heatmaps and bar charts) for deeper insights.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Jupyter Notebook](#jupyter-notebook)
   - [API](#api)
5. [API Endpoints](#api-endpoints)
6. [Key Components](#key-components)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Visualization](#visualization)
9. [LangFuse Integration](#langfuse-integration)
10. [Project Structure](#project-structure)
11. [Contributing](#contributing)
12. [License](#license)

## Project Overview

The project focuses on:

1. Loading and preprocessing documents.
2. Generating a test dataset.
3. Evaluating the RAG system using RAGAS metrics.
4. Visualizing the evaluation results.
5. Tracking performance metrics using LangFuse.

It supports both local evaluation via a Jupyter notebook and API interaction via a FastAPI service.

## Prerequisites

- Python 3.7+
- OpenAI API key (for language detection)
- LangFuse account (for performance tracking)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/rag-evaluation-project.git
   cd rag-evaluation-project
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file:

   ```bash
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

   ```bash
   jupyter notebook
   ```

3. Open and run `ragas.ipynb`.

The notebook guides you through the process of loading data, generating a test dataset, evaluating the RAG system, and visualizing the results.

### API

1. Ensure your document content is in `bhz_content.txt`.
2. Start the FastAPI server:

   ```bash
   python api.py
   ```

3. The API will be available at `http://localhost:8001`.

## API Endpoints

### POST /evaluate

This endpoint evaluates the RAG system using RAGAS metrics based on the provided document.

**Parameters:**

| Name          | Type  | Description                                    | Required |
|---------------|-------|------------------------------------------------|----------|
| file_path     | str   | The file path to the document to evaluate       | Yes      |
| test_size     | int   | Number of test questions to generate (default: 2) | No       |
| chunk_method  | int   | 1 for semantic chunking, 2 for character-based chunking | No |
| distributions | dict  | Test set distributions for different question types | No |

**Request Body:**

```json
{
  "file_path": "path/to/your/text/file.txt",
  "test_size": 10,
  "chunk_method": 1,
  "distributions": {
    "simple": 0.4,
    "reasoning": 0.2,
    "multi_context": 0.2,
    "conditional": 0.2
  }
}
```

**Response:**

```json
{
  "results": [
    {
      "question": "What is the purpose of the Ragas framework?",
      "faithfulness": 0.97,
      "answer_relevancy": 0.98,
      "context_recall": 1.0,
      "context_precision": 0.99
    },
    ...
  ],
  "langfuse_trace_id": "trace_id",
  "heatmap_image": "base64_encoded_heatmap_image",
  "bar_chart_image": "base64_encoded_bar_chart_image"
}
```

**Example Requests:**

1. Basic evaluation with default test size:

   ```bash
   curl -X POST "http://localhost:8001/evaluate" \
        -H "Content-Type: application/json" \
        -d '{"file_path": "path/to/file.txt"}'
   ```

2. Evaluation with custom parameters:

   ```bash
   curl -X POST "http://localhost:8001/evaluate" \
        -H "Content-Type: application/json" \
        -d '{"file_path": "path/to/file.txt", "test_size": 15, "chunk_method": 2}'
   ```

## Key Components

1. **Document Loading and Chunking**: The project uses LangChain’s `TextLoader` and `SemanticChunker` to load and split documents into chunks for evaluation.
2. **Language Detection**: OpenAI’s `ChatGPT` model is used for detecting the language of the document.
3. **Test Set Generation**: `TestsetGenerator` from RAGAS generates test sets for evaluation.
4. **Evaluation**: Evaluation of RAG systems using RAGAS metrics, such as **faithfulness**, **answer relevancy**, **context recall**, and **context precision**.
5. **Visualization**: Seaborn and Matplotlib are used for generating visualizations such as heatmaps and bar charts.
6. **Performance Tracking**: All results are logged and tracked using **LangFuse**.

## Evaluation Metrics

The project evaluates the following metrics:

1. **Faithfulness**: Measures how accurate the answer is based on the retrieved context.
2. **Answer Relevancy**: Measures the relevance of the answer to the given question.
3. **Context Recall**: Measures how much of the relevant context is recalled by the RAG system.
4. **Context Precision**: Measures how precise the context is in relation to the question.

## Visualization

The system generates visualizations (heatmaps and bar charts) to help you visualize the evaluation metrics for each question in the test set.

- **Heatmap**: Displays the evaluation of each metric.
- **Bar Chart**: Shows answer relevancy for each question.

In the notebook, the images are displayed inline, while in the API, they are returned as base64-encoded strings.

## LangFuse Integration

LangFuse is used to track and monitor the performance of the RAG system. It provides detailed logging of evaluation metrics for each test case, ensuring transparency and enabling deeper insights into how the RAG system performs.

## Project Structure

- `ragas.ipynb`: Jupyter notebook for running RAG evaluations interactively.
- `api.py`: FastAPI application for RAG evaluation.
- `app/`: Directory containing all modules related to evaluation and test set generation.
- `bhz_content.txt`: Sample source document used for RAG evaluation.
- `requirements.txt`: Python dependencies.
- `.env`: Environment variables file (not included in the repository).

## Contributing

We welcome contributions! If you have any ideas or improvements, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or inquiries, please reach out to us at [sebastian.schkudlara@gmail.com].
