from pydantic import BaseModel
from typing import Optional, Dict, List

class EvaluationRequest(BaseModel):
    file_path: str
    test_size: int = 2
    chunk_method: int = 1
    distributions: Dict[str, float] = {
        "simple": 0.4,
        "reasoning": 0.2,
        "multi_context": 0.2,
        "conditional": 0.2
    }

class EvaluationResponse(BaseModel):
    results: List[Dict[str, float]]
    langfuse_trace_id: Optional[str]
    heatmap_image: str
    bar_chart_image: str
