from langfuse import Langfuse
import os

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

def log_to_langfuse(evaluation_result, file_path):
    trace = langfuse.trace(
        name="Evaluation Trace",
        user_id="user_id_placeholder",
        metadata={"source_file": file_path},
        tags=["evaluation"]
    )
    
    for row in evaluation_result:
        for metric_name in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            langfuse.score(
                name=metric_name,
                value=row[metric_name],
                trace_id=trace.id
            )
    
    return trace
