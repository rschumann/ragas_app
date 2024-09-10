# app/services.py

from app.models import EvaluationRequest
from app.document_processing import load_and_chunk_documents, detect_language
from app.testset_generator import generate_testset
from app.evaluation import perform_evaluation
from app.visualization import generate_visualizations
from app.langfuse_integration import log_to_langfuse
from fastapi.responses import JSONResponse
import traceback

import logging
logger = logging.getLogger(__name__)

async def evaluate_rag(request: EvaluationRequest):
    try:
        documents = load_and_chunk_documents(request.file_path, request.chunk_method)
        detected_language = detect_language(documents[0].page_content)
        
        testset = generate_testset(documents, detected_language, request.test_size, request.distributions)
        
        evaluation_result = await perform_evaluation(testset)
        
        heatmap_img, bar_chart_img = generate_visualizations(evaluation_result)
        
        trace = log_to_langfuse(evaluation_result, request.file_path)

        return JSONResponse(content={
            "results": evaluation_result,
            "langfuse_trace_id": trace.id,
            "heatmap_image": heatmap_img,
            "bar_chart_image": bar_chart_img
        })
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
