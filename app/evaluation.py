# app/evaluation.py

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import pandas as pd
from app.utils import api_rag

async def perform_evaluation(testset):
    data = await prepare_evaluation_data(testset)
    
    dataset = Dataset.from_pandas(pd.DataFrame(data))

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
    )

    df = result.to_pandas()

    # Format the results to match the original structure
    formatted_results = []
    for _, row in df.iterrows():
        formatted_result = {
            "question": row["user_input"],
            "faithfulness": float(row["faithfulness"]),
            "answer_relevancy": float(row["answer_relevancy"]),
            "context_recall": float(row["context_recall"]),
            "context_precision": float(row["context_precision"])
        }
        formatted_results.append(formatted_result)

    return formatted_results

async def prepare_evaluation_data(testset):
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": testset.to_pandas()["ground_truth"].to_list()}
    
    for query in testset.to_pandas()["question"].to_list():
        answer, contexts = await api_rag(query)
        data["question"].append(query)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
    
    data['retrieved_contexts'] = data['contexts']
    data['user_input'] = data['question']
    data['response'] = data['answer']
    data['reference'] = data['ground_truth']

    return data
